// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "build.h"
#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE == PerfRunType::L1_TO_L1, "Only L1 to L1 is supported for this benchmark");

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

// Algorithm invariants
static_assert(FULL_CT_DIM % BLOCK_CT_DIM == 0, "FULL_CT_DIM must be divisible by BLOCK_CT_DIM");

// Test assumptions
// static_assert(FULL_RT_DIM * FULL_CT_DIM == params->TILE_CNT, "FULL_RT_DIM * FULL_CT_DIM must be equal to params->TILE_CNT");

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "llk_unpack_untilize.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    constexpr std::uint32_t TILE_SIZE = 2048 / 16; // size of tile in 16B words

    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
        _llk_unpack_untilize_init_(formats.unpack_dst, TILE_SIZE, FACE_R_DIM);
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")

        for (std::uint32_t tile = 0; tile < params->TILE_CNT; tile += FULL_CT_DIM)
        {
            _llk_unpack_untilize_pass_<true>(PERF_ADDRESS(PERF_INPUT_A, tile), FULL_CT_DIM);
            _llk_unpack_untilize_pass_<false>(PERF_ADDRESS(PERF_INPUT_A, tile), FULL_CT_DIM);
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")

#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(4, formats.math);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#endif
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")

        for (std::uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
        {
            for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                {
                    LLK_ASSERT(
                        (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles");
                    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                        block_tile, formats.math, formats.math);
                }
                _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    constexpr bool UNTILIZE = false;

    {
        ZONE_SCOPED("INIT")

#ifdef ARCH_BLACKHOLE
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
        _llk_pack_init_<UNTILIZE, false, false>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
#else
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, UNTILIZE>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
        _llk_pack_init_<UNTILIZE, false>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>();
#endif
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")

        for (std::uint32_t loop = 0; loop < params->LOOP_FACTOR; loop++)
        {
            for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
            {
                std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                _llk_packer_wait_for_math_done_();
                for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                {
                    LLK_ASSERT(
                        (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                        "Block tile index exceeds maximum destination tiles");
                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, UNTILIZE>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                }
                _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            }
        }
        PROFILER_SYNC();
    }
}

#endif
