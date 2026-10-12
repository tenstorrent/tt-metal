// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context                          = 0;
std::uint32_t pack_sync_tile_dst_ptr                   = 0;
std::uint32_t math_sync_tile_dst_index                 = 0;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
static constexpr std::uint32_t MAX_TILES_DEST          = ckernel::get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>();
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

// binop_with_scalar is float-only (add/sub/mul/div/rsub). All tested formats
// (Float32, Float16_b, Bfp8_b) pass through untouched; anything unexpected
// falls back to a raw 16-bit (UInt16) move so the harness still links.
static constexpr std::uint8_t resolve_binop_format(std::uint8_t in)
{
    switch (static_cast<DataFormat>(in))
    {
        case DataFormat::Float32:
        case DataFormat::Bfp8_b:
        case DataFormat::Float16_b:
            return in;
        default:
            return static_cast<std::uint8_t>(DataFormat::UInt16);
    }
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
#endif
    const std::uint8_t UNPACK_FMT = resolve_binop_format(UNPACK_A_IN);

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES, UNPACK_TRANSPOSE_WITHIN_FACE, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), UNPACK_FMT, UNPACK_FMT);
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
                _perf_unpack_loop_set_valid<true /* set_a */, is_fp32_dest_acc_en /* set_b */, PERF_RUN_TYPE>(num_faces * TILE_CNT * LOOP_FACTOR /* iterations */);
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest */, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[tile]), UNPACK_FMT, UNPACK_FMT);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

#include "llk_sfpu/ckernel_sfpu_binop_with_unary.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    const std::uint8_t MATH_FMT = resolve_binop_format(UNPACK_A_IN);
    const bool is_int_fpu_en    = false;

    auto run_scalar_binop = [](std::uint32_t dst_index)
    {
        // VectorMode::RC drives four faces (eight rows each), matching the
        // production add_unary_tile / sub_unary_tile / ... APIs.
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            calculate_binop_with_scalar,
            (APPROX_MODE, SFPU_BINOP_MODE, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
            dst_index,
            VectorMode::RC,
            SFPU_UNARY_SCALAR);
    };

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), MATH_FMT);
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE && perf_mock_mop<PERF_RUN_TYPE>() && !unpack_to_dest) // Wormhole: one MOP stream (perf.h)
        {
            _perf_math_loop_clear_valid<true, true, PERF_RUN_TYPE>(num_faces * TILE_CNT * LOOP_FACTOR);
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (unpack_to_dest)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            tile % MAX_TILES_DEST, MATH_FMT, MATH_FMT);
                    }
                    else
                    {
                        _perf_math_loop_clear_valid</* clear A */ true, /* clear B */ true>(num_faces);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, MATH_FMT, MATH_FMT);
                        }
                        else
                        {
                            _perf_math_loop_clear_valid</* clear A */ true, /* clear B */ true>(num_faces);
                        }
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (!unpack_to_dest)
                        {
                            _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, MATH_FMT, MATH_FMT);
                        }
                        run_scalar_binop(block_tile);
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, MATH_FMT, MATH_FMT);
                        // SFPU_UNARY_CALL rebases DEST through _llk_math_eltwise_sfpu_start_
                        // for each tile; datacopy_uninit is a no-op on Wormhole/Blackhole.
                        run_scalar_binop(block_tile);
                    }
                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();
        PROFILER_SYNC();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_Res = params.buffer_Res;
#endif
    const std::uint8_t PACK_FMT = resolve_binop_format(UNPACK_A_IN);

    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, FACE_R_DIM * FACE_C_DIM * num_faces /* tile_size */);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
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
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT((block_tile < MAX_TILES_DEST), "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT((block_tile < MAX_TILES_DEST), "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
