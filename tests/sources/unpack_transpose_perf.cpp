// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static_assert(PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE, "Math isolation not supported for unpack_transpose");
static_assert(PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE, "Pack isolation not supported for unpack_transpose");
static_assert(PERF_RUN_TYPE != PerfRunType::L1_CONGESTION, "L1 congestion not supported for unpack_transpose");

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_src, formats.unpack_src, formats.unpack_dst, formats.unpack_dst, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES);
        _llk_unpack_A_init_<>(
            params->UNPACK_TRANSPOSE_FACES, params->UNPACK_TRANSPOSE_WITHIN_FACE, FACE_R_DIM, TILE_NUM_FACES, formats.unpack_src, formats.unpack_dst);
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")

        for (std::uint32_t tile = 0; tile < params->TILE_CNT; tile++)
        {
            _llk_unpack_A_<>(PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_src, formats.unpack_dst);
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(TILE_NUM_FACES, formats.math);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(TILE_NUM_FACES, formats.math);
#endif
        PROFILER_SYNC();
    }

    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            // _llk_unpack_A     sets both A and B valid
            return _perf_math_loop_clear_valid<true, true>(params->TILE_CNT * TILE_NUM_FACES);
        }

        for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
        {
            std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

            _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
            for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
            {
                LLK_ASSERT(
                    (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles");
                _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                    block_tile, formats.math, formats.math);
            }
            _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
    {
        ZONE_SCOPED("INIT")

        _llk_pack_hw_configure_<is_fp32_dest_acc_en>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT);
        _llk_pack_init_<false, false>(formats.pack_dst);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            return;
        }

        for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
        {
            std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

            _llk_packer_wait_for_math_done_();
            for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
            {
                LLK_ASSERT(
                    (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                    "Block tile index exceeds maximum destination tiles");
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
            }
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        }
        PROFILER_SYNC();
    }
}

#endif
