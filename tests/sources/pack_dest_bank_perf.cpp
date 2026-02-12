
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>

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

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_src,
            formats.unpack_src,
            formats.unpack_dst,
            formats.unpack_dst,
            FACE_R_DIM,
            FACE_R_DIM,
            /* num_faces */ 4,
            /* num_faces */ 4);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0, 0, FACE_R_DIM, 4, formats.unpack_src, formats.unpack_dst);
        PROFILER_SYNC();

        for (std::uint32_t tile = 0; tile < params->TILE_CNT; tile++)
        {
            _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_src, formats.unpack_dst);
        }
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return _perf_unpack_loop_set_valid<true, true>(params->TILE_CNT * TILE_NUM_FACES * LOOP_FACTOR);
        }
        else
        {
            for (int loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < params->TILE_CNT; tile++)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, tile), formats.unpack_src, formats.unpack_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#ifdef FORMAT_INT32
const bool is_int_fpu_en = true;
#else
const bool is_int_fpu_en = false;
#endif

#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

void run_kernel(const volatile struct RuntimeParams* params)
{
    {
        ZONE_SCOPED("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, is_int_fpu_en>(4, formats.math);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en>(4, formats.math);
#endif

        for (std::uint32_t block_tile = 0; block_tile < params->TILE_CNT; block_tile++)
        {
            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                block_tile, formats.math, formats.math, 4);
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            return _perf_math_loop_clear_valid<true, true>(params->TILE_CNT * TILE_NUM_FACES * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (int loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
#ifdef ARCH_BLACKHOLE
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math, 4);
#else
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
#endif
                    }
                }
            }
        }
        else
        {
            for (int loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
#ifdef ARCH_BLACKHOLE
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math, 4);
#else
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
#endif
                    }
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
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
    {
        ZONE_SCOPED("INIT")
#ifdef ARCH_BLACKHOLE
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT, FACE_R_DIM, TILE_C_DIM, 4);
        _llk_pack_init_<false, false, false>(formats.pack_src, formats.pack_dst, FACE_R_DIM, TILE_C_DIM, params->num_faces, false, false, params->TILE_CNT);
        reconfigure_packer_l1_acc(params->L1_ACC);
#else
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, TILE_WIDTH * TILE_HEIGHT, FACE_R_DIM, 4);
        _llk_pack_init_<false, false>(formats.pack_dst, FACE_R_DIM, 4);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>();
#endif
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_packer_wait_for_math_done_();
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            return;
        }
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (int loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < params->TILE_CNT; tile++)
                {
                    // Left in a loop here since perf measurements are dividing with this TILE_CNT also

                    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(tile, PERF_ADDRESS(PERF_OUTPUT, tile));
                }
            }
        }
        else
        {
            for (int loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < params->TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    std::uint32_t block_tiles = std::min(params->TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
