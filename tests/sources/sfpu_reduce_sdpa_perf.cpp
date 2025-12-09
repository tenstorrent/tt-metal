// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
uint32_t unp_cfg_context          = 0;
uint32_t pack_sync_tile_dst_ptr   = 0;
uint32_t math_sync_tile_dst_index = 0;

static constexpr uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        // Configure unpacker for Float16_b format
        _llk_unpack_A_hw_configure_<is_fp32_dest_acc_en, StochRndType::None>(formats.unpack_src, formats.unpack_dst, FACE_R_DIM, 0, 4);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0, 0, FACE_R_DIM, 4, formats.unpack_src, formats.unpack_dst);
        PROFILER_SYNC();
    }
    {
        ZONE_SCOPED("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            return;
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Set valid for source A only (B is not used in this operation)
            return _perf_unpack_loop_set_valid<true, false>(TILE_CNT * LOOP_FACTOR);
        }
        else
        {
            for (uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, i), formats.unpack_src, formats.unpack_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel()
{
    constexpr uint32_t block_height = BLOCK_RT_DIM;

    {
        ZONE_SCOPED("INIT")
        // Initialize datacopy from srcA to dest
#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(4, formats.math);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(4, formats.math);
#endif
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<false, false>(formats.math, formats.math);

        // Initialize SFPU for reduce operation
        _llk_math_eltwise_unary_sfpu_init_<SfpuType::reduce>();

        // Initialize SDPA reduce using unified function
        _init_reduce_<PoolType::MAX, DataFormat::Float16_b>(BLOCK_CT_DIM);

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
            // Clear valid for source A only (B is not used)
            return _perf_math_loop_clear_valid<true, false>(TILE_CNT * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(0);
            // For MATH_ISOLATE, we need to properly handle data valid flags
            // The unpack thread sets valid flags, and we need to clear them
            for (uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    // Wait for source A to be valid
                    // TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::SRCA_VLD);

                    // Assume data is already in dest registers (skipping A2D copy)
                    // Run the SFPU reduce SDPA calculation
                    // This is the core computation we want to measure

                    _calculate_reduce_<PoolType::MAX, REDUCE_COL, DataFormat::Float16_b>(block_height);

                    // Clear the valid flag for source A
                    TTI_CLEARDVALID(1, 0);
                }
            }

            _llk_math_eltwise_unary_sfpu_done_();
        }
        else
        {
            // Full L1-to-L1 operation
            for (uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    // Wait for destination to be available
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();

                    // Copy from srcA to dest
                    for (uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            block_start + block_tile, formats.math, formats.math);
                    }

                    // Start SFPU operation
                    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(0);

                    // Call the SFPU SDPA reduce function
                    constexpr uint32_t block_height = BLOCK_RT_DIM;
                    _calculate_reduce_<PoolType::MAX, REDUCE_COL, DataFormat::Float16_b>(block_height);

                    _llk_math_eltwise_unary_sfpu_done_();
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

void run_kernel()
{
    {
        ZONE_SCOPED("INIT")
        // Configure packer hardware
#ifdef ARCH_BLACKHOLE
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#else
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(formats.pack_src, formats.pack_dst, 16 * 16 * 4);
#endif

        _llk_pack_init_<false, false, DstTileFaceLayout::RowMajor, false>(formats.pack_dst);

        // Initialize destination for packing
#ifdef ARCH_BLACKHOLE
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileFaceLayout::RowMajor>();
#else
        _llk_pack_dest_init_<DstSync::SyncHalf, false, DstTileFaceLayout::RowMajor, false>();
#endif
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
            for (uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                }
            }
        }
        else
        {
            // Full L1-to-L1 operation
            for (uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, false>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
