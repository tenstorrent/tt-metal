// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Wormhole metal relu_clamp_int / relu_clamp_uint (the Int32 / UInt32 relu_min_tile_* and
// relu_max_tile_* kernels), run through the production init and SFPU_UNARY_CALL over every tile.
//
// RELU_CLAMP_VARIANT: 0 int relu_min, 1 int relu_max, 2 uint relu_min, 3 uint relu_max.
// RELU_CLAMP_THRESHOLD carries the raw 32-bit threshold word. With MM_THROTTLE > 0 a throttled
// matmul's ADDR_MOD_6 is written between the init and the first body, so a body that still
// advances Dest through ADDR_MOD_6 stores every row to the same place.

#include <algorithm>
#include <cstdint>

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

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#ifndef SPEED_OF_LIGHT
    const Operand& buffer_A = params.buffer_A;
#endif
    constexpr auto UNPACK_FMT = UNPACK_A_IN;

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
                _perf_unpack_stream_set_valid<true /* set_a */, is_fp32_dest_acc_en /* set_b */, PERF_RUN_TYPE>(
                    num_faces * TILE_CNT * LOOP_FACTOR /* iterations */);
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

// clang-format off
// The init header forward-declares relu_*_init; after their definitions that is -Wredundant-decls.
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_init.h"
#include "llk_sfpu/ckernel_sfpu_relu.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
// clang-format on

static constexpr bool RELU_CLAMP_IS_LOWER_BOUND = RELU_CLAMP_VARIANT == 0 || RELU_CLAMP_VARIANT == 2;
static constexpr bool RELU_CLAMP_IS_UNSIGNED    = RELU_CLAMP_VARIANT >= 2;
static_assert(RELU_CLAMP_VARIANT >= 0 && RELU_CLAMP_VARIANT <= 3, "RELU_CLAMP_VARIANT must be 0..3");

static void relu_clamp_init()
{
    if constexpr (RELU_CLAMP_IS_LOWER_BOUND)
    {
        llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, is_fp32_dest_acc_en>();
    }
    else
    {
        llk_math_eltwise_unary_sfpu_init<SfpuType::relu_max, is_fp32_dest_acc_en>();
    }
#if defined(MM_THROTTLE) && MM_THROTTLE > 0
    // The throttled matmul MOP's ADDR_MOD_6: no Dest advance.
    ckernel::addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0, .clr = 1}}.set(ADDR_MOD_6);
#endif
}

static void relu_clamp_tile(const std::uint32_t dst_index)
{
    if constexpr (RELU_CLAMP_IS_UNSIGNED)
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            relu_clamp_uint,
            (APPROX_MODE, RELU_CLAMP_IS_LOWER_BOUND, DataFormat::UInt32, 8 /* ITERATIONS */),
            dst_index,
            VectorMode::RC,
            RELU_CLAMP_THRESHOLD);
    }
    else
    {
        SFPU_UNARY_CALL(
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            relu_clamp_int,
            (APPROX_MODE, RELU_CLAMP_IS_LOWER_BOUND, 8 /* ITERATIONS */),
            dst_index,
            VectorMode::RC,
            RELU_CLAMP_THRESHOLD);
    }
}

void run_kernel(RUNTIME_PARAMETERS)
{
    constexpr auto MATH_FMT      = UNPACK_A_IN;
    constexpr bool is_int_fpu_en = false;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), MATH_FMT);
        relu_clamp_init();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE && perf_mock_mop<PERF_RUN_TYPE>() && !unpack_to_dest) // Wormhole: one MOP stream
                                                                                                                              // (perf.h)
        {
            _perf_math_stream_clear_valid<true, true, PERF_RUN_TYPE>(num_faces * TILE_CNT * LOOP_FACTOR);
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
                        relu_clamp_tile(block_tile);
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
                        relu_clamp_tile(block_tile);
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
    constexpr auto PACK_FMT = UNPACK_A_IN;

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
        else if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                    if constexpr (PERF_PACK_SECTION_RELEASE)
                    {
                        _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
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
