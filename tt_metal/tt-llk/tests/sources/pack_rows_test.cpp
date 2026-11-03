// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "counters.h"
#include "llk_defs.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
#endif
    const int num_total_tiles = NUM_TILES_IN_BLOCK * NUM_BLOCKS;
    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /*num_faces*/, 4 /*num_faces*/);
        _llk_unpack_A_init_<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_loop_set_valid<true /* src A */, is_fp32_dest_acc_en>(LOOP_FACTOR * num_total_tiles * 4);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int tile = 0; tile < num_total_tiles; ++tile)
                {
                    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(params.buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_lib_math_wrappers.h"
#include "params.h"

using namespace ckernel;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
#endif
    const bool is_int_fpu_en = false;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
            ckernel::DEFAULT_TENSOR_SHAPE, formats.math);
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            if constexpr (unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t block = 0; block < num_blocks; ++block)
                    {
                        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }
                }
            }
            else
            {
                _perf_math_loop_clear_valid<true /* src A */, true /* src B */>(LOOP_FACTOR * num_blocks * num_tiles_in_block * 4);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            if constexpr (!unpack_to_dest)
            {
                for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
                {
                    for (std::uint32_t block = 0; block < num_blocks; ++block)
                    {
                        for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                    {
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                            tile, formats.math, formats.math);
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

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "llk_pack_rows.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_blocks         = params.NUM_BLOCKS;
    const std::uint32_t num_tiles_in_block = params.NUM_TILES_IN_BLOCK;
#endif
    const bool UNTILIZE = false;

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * TILE_NUM_FACES);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, llk_test_pack_mode_v<UNTILIZE, false>>();
        _llk_pack_rows_init_(params.NUM_ROWS_TO_PACK);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
        }
        else
        {
            constexpr bool sync_with_math = PERF_RUN_TYPE == PerfRunType::L1_TO_L1;
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block = 0; block < num_blocks; ++block)
                {
                    if constexpr (sync_with_math)
                    {
                        _llk_packer_wait_for_math_done_();
                    }
                    for (std::uint32_t tile = 0; tile < num_tiles_in_block; ++tile)
                    {
                        std::uint32_t res_tile_idx = (block * num_tiles_in_block) + tile;
                        LLK_ASSERT(
                            (tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "Block tile index exceeds maximum destination tiles");
                        _llk_pack_rows_(tile, L1_ADDRESS(params.buffer_Res[res_tile_idx]));
                    }
                    if constexpr (sync_with_math)
                    {
                        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_pack_rows_uninit_();
}

#endif
