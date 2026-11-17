// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "experimental/llk_unpack_A_custom.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

inline void unpack_tiles(const std::uint32_t tile_cnt, const Operand& buffer_A)
{
    for (std::uint32_t tile = 0; tile < tile_cnt; ++tile)
    {
        _llk_unpack_A_custom_(L1_ADDRESS(buffer_A[tile]));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor = params.LOOP_FACTOR;
    const std::uint32_t tile_cnt    = params.TILE_CNT;
    const Operand& buffer_A         = params.buffer_A;
#else
    constexpr std::uint32_t loop_factor = LOOP_FACTOR;
    constexpr std::uint32_t tile_cnt    = TILE_CNT;
#endif
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
            _perf_unpack_stream_set_valid<true /*set_a*/, false /*set_b*/, PERF_RUN_TYPE>(loop_factor * tile_cnt);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                unpack_tiles(tile_cnt, buffer_A);
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "experimental/llk_math_eltwise_unary_datacopy_custom.h"
#include "llk_math_common.h"
#include "params.h"

inline void math_block(const std::uint32_t num_tiles_in_block)
{
    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
    {
        _llk_math_eltwise_unary_datacopy_custom_(tile);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
#else
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
#endif
    const std::uint32_t tile_cnt = num_blocks * num_tiles_in_block;
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
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            _perf_math_stream_clear_valid<true /*clear_a*/, false /*clear_b*/, PERF_RUN_TYPE>(loop_factor * tile_cnt);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
                    math_block(num_tiles_in_block);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
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
#include "params.h"

inline void pack_block(const std::uint32_t block, const std::uint32_t num_tiles_in_block, const Operand& buffer_Res)
{
    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
    {
        const std::uint32_t result_tile = block * num_tiles_in_block + tile;
        _llk_pack_<DstSync::SyncHalf, false, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[result_tile]));
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t loop_factor        = params.LOOP_FACTOR;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
    const Operand& buffer_Res              = params.buffer_Res;
#else
    constexpr std::uint32_t loop_factor        = LOOP_FACTOR;
    constexpr std::uint32_t num_blocks         = NUM_BLOCKS;
    constexpr std::uint32_t num_tiles_in_block = NUM_TILES_IN_BLOCK;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<false /* is_fp32_dest_acc_en */, PackMode::Default>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */);
        _llk_pack_dest_init_<DstSync::SyncHalf, false>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < loop_factor; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    pack_block(block, num_tiles_in_block, buffer_Res);
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}
#endif
