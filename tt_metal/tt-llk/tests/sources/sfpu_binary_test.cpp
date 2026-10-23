// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

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
    // Compile-time formats can put each operand in its own buffer. Runtime formats
    // stay on the interleaved buffer_A path.
#ifndef RUNTIME_FORMATS
    constexpr bool mixed_src_formats = formats.unpack_A_src != formats.unpack_B_src;
#else
    constexpr bool mixed_src_formats = false;
#endif
    // A 32-bit mixed pair hangs: unpack-to-dest waits on a handshake the B unpack skips.
    static_assert(!(mixed_src_formats && unpack_to_dest), "mixed src formats cannot unpack to dest");
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    const Operand& buffer_A                = params.buffer_A;
    const Operand& buffer_B                = params.buffer_B;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
#endif
    // Mocks post the same src dvalids per tile as _llk_unpack_A_.
    const std::uint32_t tile_iters = LOOP_FACTOR * TILE_CNT;

    {
        START_PERF_MEASURE("INIT")
        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);
        _llk_unpack_A_init_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            0 /* transpose_of_faces */,
            0 /* within_face_16x16_transpose */,
            ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, num_faces),
            formats.unpack_A_src,
            formats.unpack_A_dst);
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")
        if constexpr (PERF_RUN_TYPE == PerfRunType::PACK_ISOLATE)
        {
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            // When unpack_to_dest is set, dest is assumed ready and math does not wait on src dvalids.
            if constexpr (!unpack_to_dest)
            {
                if constexpr (BROADCAST_TYPE == BroadcastType::NONE)
                {
                    // SrcA plus a SrcB zerosrc dvalid every face, including dest_acc=No (tenstorrent/budabackend#1230).
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(tile_iters * num_faces /*iterations*/);
                }
                else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
                {
                    _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(tile_iters * num_faces /*iterations*/);
                }
                else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
                {
                    // Interleave per tile; bulk posting deadlocks on the 2-deep src banks.
                    for (std::uint32_t i = 0; i < tile_iters; ++i)
                    {
                        _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(1 /*iterations*/);
                        _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(1 /*iterations*/);
                    }
                }
                else
                {
                    _perf_unpack_loop_set_valid<true /*set_a*/, true /*set_b*/>(tile_iters /*iterations*/);
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                if constexpr (mixed_src_formats)
                {
                    // Src is 2-deep, so the two operands stay in lockstep. Reconfig
                    // switches the L1 format and the tile-size GPR between them.
                    for (std::uint32_t pair = 0; pair < TILE_CNT / 2; ++pair)
                    {
                        _llk_unpack_A_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(buffer_A[pair]), formats.unpack_A_src, formats.unpack_A_dst);
                        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE>(
                            formats.unpack_B_src, formats.unpack_B_dst, TILE_SIZE_UNPACK_B);
                        _llk_unpack_A_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(buffer_B[pair]), formats.unpack_B_src, formats.unpack_B_dst);
                        _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE>(
                            formats.unpack_A_src, formats.unpack_A_dst, TILE_SIZE_UNPACK_A);
                    }
                }
                else
                {
                    for (std::uint32_t i = 0; i < TILE_CNT; i++)
                    {
                        _llk_unpack_A_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
                            L1_ADDRESS(buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
                    }
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_unpack_A_uninit_<BROADCAST_TYPE>();
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_defs.h"
#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "params.h"
#include "sfpu_operations.h"

// Fresh semantic bodies: keep after sfpu_operations.h in their own include
// block (their templates consume its transitive typed helpers at
// definition-time lookup; clang-format sorts blocks independently).
#include "fresh_cpp/isclose.h"
#include "fresh_cpp/lcm.h"
#include "fresh_cpp_operations.h"
// Storm-contract canonical per-op semantic bodies (new bodies never land in
// the legacy aggregator above).
#include "fresh_cpp/shift.h"
// Storm-contract semantic bodies (one op per header, fresh_cpp/README.md).
#include "fresh_cpp/atan2.h"
#include "fresh_cpp/atan2_fitted.h"
#include "fresh_cpp/binary_float.h"
#include "fresh_cpp/binarybitwise.h"
#include "fresh_cpp/binarycomp.h"
#include "fresh_cpp/binaryfmod.h"
#include "fresh_cpp/binarypow.h"
#include "fresh_cpp/binaryremainder.h"

#ifndef FRESH_CPP_IMPL
#define FRESH_CPP_IMPL 0
#endif

using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
#ifndef RUNTIME_FORMATS
    constexpr bool mixed_src_formats = formats.unpack_A_src != formats.unpack_B_src;
#else
    constexpr bool mixed_src_formats = false;
#endif
    // A 32-bit mixed pair hangs: unpack-to-dest waits on a handshake the B unpack skips.
    static_assert(!(mixed_src_formats && unpack_to_dest), "mixed src formats cannot unpack to dest");
#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR        = params.LOOP_FACTOR;
    const std::uint32_t num_faces          = params.num_faces;
    const int NUM_BLOCKS                   = params.NUM_BLOCKS;
    const std::uint32_t NUM_TILES_IN_BLOCK = params.NUM_TILES_IN_BLOCK;
    const std::uint32_t TILE_CNT           = params.TILE_CNT;
    // --speed-of-light emits ZERO_POINT as a constexpr and leaves RuntimeParams empty.
    const std::uint32_t ZERO_POINT = params.ZERO_POINT;
#endif
    const bool is_int_fpu_en         = false;
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;
    const std::uint32_t tile_iters   = LOOP_FACTOR * TILE_CNT;
    // Mirrors production: reconfig SrcA between the two datacopies.
    [[maybe_unused]] auto copy_mixed_pair = [&](std::uint32_t tile)
    {
        _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(tile, formats.math, formats.math);
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en>(formats.unpack_B_dst);
        _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
            tile + 1, formats.math, formats.math);
        _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en>(formats.unpack_A_dst);
    };

    {
        START_PERF_MEASURE("INIT")
        _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
        _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
            num_faces, formats.math);
        test_utils::call_binary_sfpu_operation_init<APPROX_MODE, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, ITERATIONS, formats.math, SFPU_SIGN_MAGNITUDE>(
            ZERO_POINT);
        if constexpr (FRESH_CPP_IMPL == 3 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
        {
            ckernel::sfpu::sfpu_binary_pow_init<APPROX_MODE>();
        }
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
                    for (int block = 0; block < NUM_BLOCKS; ++block)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            // Handshake only. Column/scalar unpack-to-dest expands the broadcast
                            // in this call, which would land in the unpack number. L1_TO_L1 below
                            // keeps BROADCAST_TYPE.
                            _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }
                }
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::NONE)
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(tile_iters * num_faces /*iterations*/);
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::ROW)
            {
                _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(tile_iters * num_faces /*iterations*/);
            }
            else if constexpr (BROADCAST_TYPE == BroadcastType::COL)
            {
                for (std::uint32_t i = 0; i < tile_iters; ++i)
                {
                    _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(1 /*iterations*/);
                    _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(1 /*iterations*/);
                }
            }
            else
            {
                _perf_math_loop_clear_valid<true /*clear_a*/, true /*clear_b*/>(tile_iters /*iterations*/);
            }
        }
        else if constexpr (PERF_RUN_TYPE == PerfRunType::MATH_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    if constexpr (mixed_src_formats)
                    {
                        for (std::uint32_t pair = 0; pair < NUM_TILES_IN_BLOCK / 2; ++pair)
                        {
                            copy_mixed_pair(pair * 2);
                        }
                    }
                    else if constexpr (!unpack_to_dest)
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }

                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; tile += 2)
                    {
                                    if constexpr (
                FRESH_CPP_IMPL == 1 &&
                (SFPU_BINARY_OPERATION == ckernel::BinaryOp::MAX || SFPU_BINARY_OPERATION == ckernel::BinaryOp::MIN))
            {
                constexpr bool is_max = SFPU_BINARY_OPERATION == ckernel::BinaryOp::MAX;
                call_binary_max_min_fresh_cpp<
                    DstSync::SyncHalf, is_fp32_dest_acc_en, is_max, 8>(
                    tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && (SFPU_BINARY_OPERATION == ckernel::BinaryOp::ADD || SFPU_BINARY_OPERATION == ckernel::BinaryOp::SUB) &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the sign-magnitude Int32
                // add/sub production path (_add_int_/_sub_int_ SIGN_MAGNITUDE).
                constexpr bool is_add = SFPU_BINARY_OPERATION == ckernel::BinaryOp::ADD;
                call_add_sub_int_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, is_add, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::MUL_INT32 &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 multiply production
                // path (metal mul_int32 SFPLOADMACRO kernel).
                call_mul_int_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 2 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::MUL_INT32 &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Fresh typed-C++ contract-domain limb-2 Int32 multiply (lane GG:
                // operands < 2^23 per the row's stimuli contract; certified
                // laneGG-evidence-20260824/mulint32_limb2_cert.c).
                call_mul_int_limb2_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::LSHFT &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 binary left shift
                // production path (metal ckernel_sfpu_shift.h raw-TTI kernel).
                call_left_shift_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && (SFPU_BINARY_OPERATION == ckernel::BinaryOp::RSHFT || SFPU_BINARY_OPERATION == ckernel::BinaryOp::LOGICAL_RSHFT) &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 binary right shift
                // production paths (metal ckernel_sfpu_shift.h raw-TTI kernels;
                // LOGICAL selects zero fill vs the arithmetic sign fill).
                constexpr bool is_logical = SFPU_BINARY_OPERATION == ckernel::BinaryOp::LOGICAL_RSHFT;
                call_right_shift_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, is_logical, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::LCM &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 lcm production
                // path (metal ckernel_sfpu_lcm.h raw-TTI + REPLAY kernel).
                call_lcm_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ISCLOSE)
            {
                // Test-only fresh typed-C++ leg for isclose.  The tolerance
                // scalars and EQUAL_NAN mirror the production dispatch
                // (sfpu_operations.h ISCLOSE branch: torch defaults rtol=1e-5,
                // atol=1e-8, EQUAL_NAN=false) so both legs compute the
                // identical operation.
                call_isclose_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, /*EQUAL_NAN=*/false, 8>(
                    tile, tile + 1, tile, VectorMode::RC, /*rtol_bits=*/0x3727c5acu, /*atol_bits=*/0x322bcc77u);
            }
            // Storm-lane S1 selectors (fresh_cpp/<op>.h semantic bodies).
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::SUB &&
                static_cast<std::uint32_t>(formats.math) != static_cast<std::uint32_t>(DataFormat::Int32))
            {
                call_binary_float_sub_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::EQ &&
                static_cast<std::uint32_t>(formats.math) != static_cast<std::uint32_t>(DataFormat::Int32))
            {
                call_binary_comp_eq_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::FMOD)
            {
                call_binary_fmod_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::REMAINDER)
            {
                call_binary_remainder_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ATAN2)
            {
                call_atan2_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            // Lane DH fitted-kernel placeholder (tt-polynomial-fitter frontier
            // atan winner + the storm-S1 quadrant fixup; provenance header in
            // fresh_cpp/atan2_fitted.h): impl 2.
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ATAN2)
            {
                call_atan2_fitted_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::BITWISE_AND)
            {
                call_binary_bitwise_and_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            // Storm S2 (agent/storm-s2): canonical fresh_cpp/<op>.h semantic bodies.
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::GCD &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 gcd production path
                // (metal ckernel_sfpu_gcd.h hand-issued REPLAY-loop kernel).
                call_gcd_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::DIV_INT32_FLOOR &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 floor-division
                // production path (metal ckernel_sfpu_div_int32_floor.h).
                call_div_int32_floor_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            // Lane EU coverage expansion (binarypow-fresh row): fresh semantic
            // pow vs the byte-untouched calculate_sfpu_binary_pow hand kernel.
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
            {
                call_binary_pow_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 3 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
            {
                SFPU_BINARY_CALL(
                    DstSync::SyncHalf,
                    is_fp32_dest_acc_en,
                    calculate_sfpu_binary_pow,
                    (APPROX_MODE, 8, is_fp32_dest_acc_en),
                    tile,
                    tile + 1,
                    tile,
                    VectorMode::RC);
            }
            else
            {
                test_utils::call_binary_sfpu_operation<DstSync::SyncHalf, is_fp32_dest_acc_en, APPROX_MODE, SFPU_BINARY_OPERATION, ITERATIONS, formats.math, SFPU_SIGN_MAGNITUDE, SFPU_DST_ROUNDING_MODE>(tile, tile + 1, tile);
            }
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
                    if constexpr (mixed_src_formats)
                    {
                        for (std::uint32_t pair = 0; pair < NUM_TILES_IN_BLOCK / 2; ++pair)
                        {
                            copy_mixed_pair(pair * 2);
                        }
                    }
                    else
                    {
                        for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                        {
                            _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                tile, formats.math, formats.math);
                        }
                    }

                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; tile += 2)
                    {
                                    if constexpr (
                FRESH_CPP_IMPL == 1 &&
                (SFPU_BINARY_OPERATION == ckernel::BinaryOp::MAX || SFPU_BINARY_OPERATION == ckernel::BinaryOp::MIN))
            {
                constexpr bool is_max = SFPU_BINARY_OPERATION == ckernel::BinaryOp::MAX;
                call_binary_max_min_fresh_cpp<
                    DstSync::SyncHalf, is_fp32_dest_acc_en, is_max, 8>(
                    tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && (SFPU_BINARY_OPERATION == ckernel::BinaryOp::ADD || SFPU_BINARY_OPERATION == ckernel::BinaryOp::SUB) &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the sign-magnitude Int32
                // add/sub production path (_add_int_/_sub_int_ SIGN_MAGNITUDE).
                constexpr bool is_add = SFPU_BINARY_OPERATION == ckernel::BinaryOp::ADD;
                call_add_sub_int_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, is_add, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::MUL_INT32 &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 multiply production
                // path (metal mul_int32 SFPLOADMACRO kernel).
                call_mul_int_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 2 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::MUL_INT32 &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Fresh typed-C++ contract-domain limb-2 Int32 multiply (lane GG:
                // operands < 2^23 per the row's stimuli contract; certified
                // laneGG-evidence-20260824/mulint32_limb2_cert.c).
                call_mul_int_limb2_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::LSHFT &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 binary left shift
                // production path (metal ckernel_sfpu_shift.h raw-TTI kernel).
                call_left_shift_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && (SFPU_BINARY_OPERATION == ckernel::BinaryOp::RSHFT || SFPU_BINARY_OPERATION == ckernel::BinaryOp::LOGICAL_RSHFT) &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 binary right shift
                // production paths (metal ckernel_sfpu_shift.h raw-TTI kernels;
                // LOGICAL selects zero fill vs the arithmetic sign fill).
                constexpr bool is_logical = SFPU_BINARY_OPERATION == ckernel::BinaryOp::LOGICAL_RSHFT;
                call_right_shift_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, is_logical, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::LCM &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 lcm production
                // path (metal ckernel_sfpu_lcm.h raw-TTI + REPLAY kernel).
                call_lcm_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ISCLOSE)
            {
                // Test-only fresh typed-C++ leg for isclose.  The tolerance
                // scalars and EQUAL_NAN mirror the production dispatch
                // (sfpu_operations.h ISCLOSE branch: torch defaults rtol=1e-5,
                // atol=1e-8, EQUAL_NAN=false) so both legs compute the
                // identical operation.
                call_isclose_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, /*EQUAL_NAN=*/false, 8>(
                    tile, tile + 1, tile, VectorMode::RC, /*rtol_bits=*/0x3727c5acu, /*atol_bits=*/0x322bcc77u);
            }
            // Storm-lane S1 selectors (fresh_cpp/<op>.h semantic bodies).
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::SUB &&
                static_cast<std::uint32_t>(formats.math) != static_cast<std::uint32_t>(DataFormat::Int32))
            {
                call_binary_float_sub_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::EQ &&
                static_cast<std::uint32_t>(formats.math) != static_cast<std::uint32_t>(DataFormat::Int32))
            {
                call_binary_comp_eq_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::FMOD)
            {
                call_binary_fmod_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::REMAINDER)
            {
                call_binary_remainder_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ATAN2)
            {
                call_atan2_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            // Lane DH fitted-kernel placeholder (tt-polynomial-fitter frontier
            // atan winner + the storm-S1 quadrant fixup; provenance header in
            // fresh_cpp/atan2_fitted.h): impl 2.
            else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ATAN2)
            {
                call_atan2_fitted_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::BITWISE_AND)
            {
                call_binary_bitwise_and_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            // Storm S2 (agent/storm-s2): canonical fresh_cpp/<op>.h semantic bodies.
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::GCD &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 gcd production path
                // (metal ckernel_sfpu_gcd.h hand-issued REPLAY-loop kernel).
                call_gcd_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (
                FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::DIV_INT32_FLOOR &&
                static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
            {
                // Test-only fresh typed-C++ leg for the Int32 floor-division
                // production path (metal ckernel_sfpu_div_int32_floor.h).
                call_div_int32_floor_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            // Lane EU coverage expansion (binarypow-fresh row): fresh semantic
            // pow vs the byte-untouched calculate_sfpu_binary_pow hand kernel.
            else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
            {
                call_binary_pow_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(tile, tile + 1, tile, VectorMode::RC);
            }
            else if constexpr (FRESH_CPP_IMPL == 3 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
            {
                SFPU_BINARY_CALL(
                    DstSync::SyncHalf,
                    is_fp32_dest_acc_en,
                    calculate_sfpu_binary_pow,
                    (APPROX_MODE, 8, is_fp32_dest_acc_en),
                    tile,
                    tile + 1,
                    tile,
                    VectorMode::RC);
            }
            else
            {
                test_utils::call_binary_sfpu_operation<DstSync::SyncHalf, is_fp32_dest_acc_en, APPROX_MODE, SFPU_BINARY_OPERATION, ITERATIONS, formats.math, SFPU_SIGN_MAGNITUDE, SFPU_DST_ROUNDING_MODE>(tile, tile + 1, tile);
            }
                    }
                    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
    _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
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
    const Operand& buffer_Res              = params.buffer_Res;
#endif
    {
        START_PERF_MEASURE("INIT")
        _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 /* tile_size */);
        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
        _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();
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
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    // The SFPU writes the result on the even dest slot. Packing operand B
                    // as well would count two packs per result. tile_cnt still counts both
                    // operands, so this is one pack per normalized pair. L1_TO_L1 below
                    // still packs every dest tile for the golden.
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; tile += 2)
                    {
                        const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[result_tile]));
                    }
                }
            }
        }
        else
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (int block = 0; block < NUM_BLOCKS; ++block)
                {
                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t tile = 0; tile < NUM_TILES_IN_BLOCK; ++tile)
                    {
                        const std::uint32_t result_tile = block * NUM_TILES_IN_BLOCK + tile;
                        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(buffer_Res[result_tile]));
                    }
                    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif
