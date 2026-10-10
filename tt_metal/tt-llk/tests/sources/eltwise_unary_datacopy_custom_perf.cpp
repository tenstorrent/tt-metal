// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Perf kernel of the Blackhole copy_tile_custom path: per iteration NUM_BLOCKS DEST sections of NUM_TILES_IN_BLOCK tiles,
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
    const Operand& buffer_A         = params.buffer_A;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
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
            _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(TILE_CNT * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_custom_(L1_ADDRESS(buffer_A[i]));
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    LLK_ASSERT(
        (NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "A DEST section holds more tiles than half of DEST");
    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(TILE_CNT * LOOP_FACTOR);
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    }
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        _llk_math_eltwise_unary_datacopy_custom_(tile);
                    }
                    if constexpr (PERF_RUN_TYPE != PerfRunType::MATH_ISOLATE)
                    {
                        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
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
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t num_faces          = params.num_faces;
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    LLK_ASSERT(
        (NUM_TILES_IN_BLOCK <= get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
        "A DEST section holds more tiles than half of DEST");
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(
            formats.pack_src, formats.pack_dst, 16 * 16 * 4 /* tile_size */, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(
                            tile, L1_ADDRESS(buffer_Res[block * NUM_TILES_IN_BLOCK + tile]));
                    }
                    if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
                    {
                        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
