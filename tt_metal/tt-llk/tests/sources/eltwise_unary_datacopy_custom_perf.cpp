// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole copy_tile_custom path: per iteration TILE_CNT tiles in DEST sections of NUM_TILES_IN_BLOCK,
// one custom unpack and one custom datacopy per tile, a standard pack per tile. One SrcA data valid per tile.

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_A_custom.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<false>(formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4, 4);
        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, false>(
            0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            _perf_unpack_loop_set_valid<true, false>(params.TILE_CNT * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
                {
                    _llk_unpack_A_custom_(L1_ADDRESS(params.buffer_A[i]));
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_unary_datacopy_custom.h"
#include "llk_math_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t tiles       = params.NUM_TILES_IN_BLOCK;
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, false>();
        _llk_math_hw_configure_<false>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true, false>(params.TILE_CNT * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < params.TILE_CNT / tiles; ++block)
                {
                    if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
                    for (std::uint32_t tile = 0; tile < tiles; ++tile)
                    {
                        _llk_math_eltwise_unary_datacopy_custom_(tile);
                    }
                    if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_dest_section_done_<DstSync::SyncHalf, false>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t tiles       = params.NUM_TILES_IN_BLOCK;
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<false, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4);
        _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4);
        _llk_pack_dest_init_<DstSync::SyncHalf, false>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < tiles; ++tile)
                {
                    _llk_pack_<DstSync::SyncHalf, false, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[tile]));
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < params.TILE_CNT / tiles; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile = 0; tile < tiles; ++tile)
                    {
                        _llk_pack_<DstSync::SyncHalf, false, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[block * tiles + tile]));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
