// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// LLK SFPU typecast kernel shared by functional and performance tests.
//
// Mirrors the production typecast compute kernel
//   ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp
// which does, per tile:  copy_tile (unpack A -> Dest) -> typecast_tile_init ->
// typecast_tile -> pack_tile.
//
// The numeric conversion is performed in-place in Dest by the SFPU
// `calculate_typecast_*` primitives, dispatched through the shared unary-SFPU
// entry points (call_unary_sfpu_operation in helpers/include/sfpu_operations.h)
// under SfpuType::typecast. Pairs realised purely by unpacker/packer format
// conversion run the same copy + pack path with no SFPU call.
//
// Compile-time configuration emitted by the Python harness:
//   TYPECAST_IN_FORMAT  : DataFormat of the L1 input (typecast IN_DTYPE)
//   TYPECAST_OUT_FORMAT : DataFormat of the L1 output (typecast OUT_DTYPE)
//   APPROX_MODE         : SFPU approximation mode
//   PERF_RUN_TYPE       : functional tests use L1_TO_L1; perf selects the measured path
//   ITERATIONS          : SFPU iteration count (typecast dispatch uses 8)
//   FAST_MODE           : typecast tests use the non-fast SFPU path
//   STABLE_SORT         : disabled for typecast
//

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

// Globals
std::uint32_t unp_cfg_context                          = 0;
std::uint32_t pack_sync_tile_dst_ptr                   = 0;
std::uint32_t math_sync_tile_dst_index                 = 0;
static constexpr ckernel::DstSync DST_SYNC             = ckernel::DstSync::SyncHalf;
static constexpr std::uint32_t MAX_TILES_DEST          = ckernel::get_dest_max_tiles<DST_SYNC, is_fp32_dest_acc_en, ckernel::DstTileShape::Tile32x32>();
static constexpr ckernel::BroadcastType BROADCAST_TYPE = ckernel::BroadcastType::NONE;

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR         = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT            = params.TILE_CNT;
    const std::uint32_t num_faces           = params.num_faces;
    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;
    const auto& buffer_A                    = params.buffer_A;
#endif
    // llk_unpack_A forbids acc_to_dest together with unpack_to_dest. Keep this
    // false so 32-bit input + dest_acc=Yes typecasts can unpack directly to Dest.
    constexpr bool UNPACK_ACC_TO_DEST = false;
    constexpr auto reuse_dest_type    = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        _llk_unpack_A_init_<BROADCAST_TYPE, UNPACK_ACC_TO_DEST, reuse_dest_type, unpack_to_dest>(
            UNPACK_TRANSPOSE_FACES,
            UNPACK_TRANSPOSE_WITHIN_FACE,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // Math-isolate assumes unpack-to-Dest data is ready and issues no handshake.
            // Otherwise set unavoidable valid bits for SrcA and, with fp32 Dest, SrcB.
            if constexpr (!unpack_to_dest)
            {
                _perf_unpack_loop_set_valid<true /*set_a*/, is_fp32_dest_acc_en /*set_b*/>(num_faces * TILE_CNT * LOOP_FACTOR /*iterations*/);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, UNPACK_ACC_TO_DEST, reuse_dest_type, unpack_to_dest>(
                        L1_ADDRESS(buffer_A[tile]), formats.unpack_A_src, formats.unpack_A_dst);
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
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

using namespace ckernel;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
#endif
    constexpr auto data_copy_type = DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        // A2D preserves raw bits for the SFPU typecast, so integer-FPU mode stays disabled.
        _llk_math_eltwise_unary_datacopy_init_wrapper_<data_copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, false /*is_int_fpu_en*/, PackMode::Default>(
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces), formats.math);
        _llk_math_pack_sync_init_<DST_SYNC, is_fp32_dest_acc_en>();

        test_utils::call_unary_sfpu_operation_init<
            SFPU_UNARY_OPERATION,
            APPROX_MODE,
            is_fp32_dest_acc_en,
            ITERATIONS,
            FAST_MODE,
            STABLE_SORT,
            false /*CLAMP_NEGATIVE*/,
            TYPECAST_IN_FORMAT,
            TYPECAST_OUT_FORMAT>(formats.math);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t tile = 0; tile < TILE_CNT; ++tile)
                {
                    if constexpr (unpack_to_dest)
                    {
                        // Unpack-to-Dest datacopy only acknowledges Dest; it copies no data.
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            tile % MAX_TILES_DEST, formats.math, formats.math);
                    }
                    else
                    {
                        // Release the SrcA/SrcB valid bits consumed by unpack.
                        _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(num_faces /*iterations*/);
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
                            // Unpack-to-Dest datacopy only acknowledges Dest; it copies no data.
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        else
                        {
                            // Release the SrcA/SrcB valid bits consumed by unpack.
                            _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(num_faces /*iterations*/);
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
                            LLK_ASSERT(block_tile < MAX_TILES_DEST, "block_tile exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        test_utils::call_unary_sfpu_operation<
                            DST_SYNC,
                            is_fp32_dest_acc_en,
                            SFPU_UNARY_OPERATION,
                            APPROX_MODE,
                            is_fp32_dest_acc_en,
                            ITERATIONS,
                            FAST_MODE,
                            STABLE_SORT,
                            false /*CLAMP_NEGATIVE*/,
                            TYPECAST_IN_FORMAT,
                            TYPECAST_OUT_FORMAT>(block_tile);
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_math_wait_for_dest_available_<DST_SYNC>();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(block_tile < MAX_TILES_DEST, "block_tile exceeds max dest tiles");
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);
                        test_utils::call_unary_sfpu_operation<
                            DST_SYNC,
                            is_fp32_dest_acc_en,
                            SFPU_UNARY_OPERATION,
                            APPROX_MODE,
                            is_fp32_dest_acc_en,
                            ITERATIONS,
                            FAST_MODE,
                            STABLE_SORT,
                            false /*CLAMP_NEGATIVE*/,
                            TYPECAST_IN_FORMAT,
                            TYPECAST_OUT_FORMAT>(block_tile);
                    }
                    _llk_math_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t TILE_CNT    = params.TILE_CNT;
    const std::uint32_t num_faces   = params.num_faces;
    const auto& buffer_Res          = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);
        _llk_pack_init_wrapper_<PackMode::Default, false /*zero_output*/>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        _llk_pack_dest_init_<DST_SYNC, is_fp32_dest_acc_en>();
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE || PERF_RUN_TYPE == PerfRunType::L1_CONGESTION)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(block_tile < MAX_TILES_DEST, "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                }
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::L1_TO_L1)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t block_start = 0; block_start < TILE_CNT; block_start += MAX_TILES_DEST)
                {
                    const std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(block_tile < MAX_TILES_DEST, "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC, is_fp32_dest_acc_en, ckernel::PackMode::Default>(block_tile, L1_ADDRESS(buffer_Res[block_start + block_tile]));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
