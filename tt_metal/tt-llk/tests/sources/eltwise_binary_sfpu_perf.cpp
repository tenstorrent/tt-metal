// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
// Globals
std::uint32_t unp_cfg_context                          = 0;
std::uint32_t pack_sync_tile_dst_ptr                   = 0;
std::uint32_t math_sync_tile_dst_index                 = 0;
static constexpr std::uint32_t MAX_TILES_DEST          = is_fp32_dest_acc_en ? 4 : 8;
static constexpr ckernel::DstSync DST_SYNC_MODE        = ckernel::DstSync::SyncHalf;
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
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;

    const std::uint32_t TILE_CNT = params.TILE_CNT;

    const bool UNPACK_TRANSPOSE_FACES       = params.UNPACK_TRANSPOSE_FACES;
    const bool UNPACK_TRANSPOSE_WITHIN_FACE = params.UNPACK_TRANSPOSE_WITHIN_FACE;
#endif

    const EltwiseBinaryReuseDestType reuse_dest_type = EltwiseBinaryReuseDestType::NONE;

    {
        START_PERF_MEASURE("INIT")

        _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
            formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, num_faces, num_faces);

        _llk_unpack_A_init_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
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
            // In case of math isolate, we don't want any software synchronization from unpack to math.
            // So we just set/clear valid bits here - which is unavoidable hardware synchronization.
            // When unpack_to_dest is used, we assume the data is immediately ready in destination register.
            // Otherwise, we assume the data is immediately ready in source A/B registers.
            if (!unpack_to_dest)
            {
                // Set valid for source A always.
                // Set valid for source B only if dest_acc is enabled.
                // Works only when unpacking to dest is not used.
                _perf_unpack_loop_set_valid<
                    /* src A */ true,
                    /* src B */ is_fp32_dest_acc_en>(
                    /* iterations*/ num_faces * TILE_CNT * LOOP_FACTOR);
            }
        }
        else if constexpr (PERF_RUN_TYPE != PerfRunType::PACK_ISOLATE) // UNPACK_ISOLATE, L1_TO_L1, L1_CONGESTION
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    _llk_unpack_A_<BROADCAST_TYPE, false /* acc_to_dest */, reuse_dest_type, unpack_to_dest>(
                        PERF_ADDRESS(PERF_INPUT_A, /* tile_idx */ i), formats.unpack_A_src, formats.unpack_A_dst);
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_datacopy.h"
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

inline void run_selected_binary_sfpu(
    std::uint32_t dst_in0, std::uint32_t dst_in1, std::uint32_t dst_out) {
    if constexpr (
        FRESH_CPP_IMPL == 1 &&
        (SFPU_BINARY_OPERATION == ckernel::BinaryOp::MAX || SFPU_BINARY_OPERATION == ckernel::BinaryOp::MIN)) {
        constexpr bool is_max = SFPU_BINARY_OPERATION == ckernel::BinaryOp::MAX;
        call_binary_max_min_fresh_cpp<
            DST_SYNC_MODE, is_fp32_dest_acc_en, is_max, 8>(
            dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 1 && (SFPU_BINARY_OPERATION == ckernel::BinaryOp::ADD || SFPU_BINARY_OPERATION == ckernel::BinaryOp::SUB) &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Test-only fresh typed-C++ leg for the sign-magnitude Int32 add/sub
        // production path (_add_int_/_sub_int_ SIGN_MAGNITUDE).
        constexpr bool is_add = SFPU_BINARY_OPERATION == ckernel::BinaryOp::ADD;
        call_add_sub_int_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, is_add, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::MUL_INT32 &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Test-only fresh typed-C++ leg for the Int32 multiply production path
        // (metal mul_int32 SFPLOADMACRO kernel).
        call_mul_int_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 2 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::MUL_INT32 &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Fresh typed-C++ contract-domain limb-2 Int32 multiply (lane GG:
        // operands < 2^23 per the row's stimuli contract; certified
        // laneGG-evidence-20260824/mulint32_limb2_cert.c).
        call_mul_int_limb2_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::LSHFT &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Test-only fresh typed-C++ leg for the Int32 binary left shift
        // production path (metal ckernel_sfpu_shift.h raw-TTI kernel).
        call_left_shift_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 1 && (SFPU_BINARY_OPERATION == ckernel::BinaryOp::RSHFT || SFPU_BINARY_OPERATION == ckernel::BinaryOp::LOGICAL_RSHFT) &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Test-only fresh typed-C++ leg for the Int32 binary right shift
        // production paths (metal ckernel_sfpu_shift.h raw-TTI kernels;
        // LOGICAL selects zero fill vs the arithmetic sign fill).
        constexpr bool is_logical = SFPU_BINARY_OPERATION == ckernel::BinaryOp::LOGICAL_RSHFT;
        call_right_shift_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, is_logical, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::LCM &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Test-only fresh typed-C++ leg for the Int32 lcm production path
        // (metal ckernel_sfpu_lcm.h raw-TTI + REPLAY kernel).
        call_lcm_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ISCLOSE)
    {
        // Test-only fresh typed-C++ leg for isclose.  Tolerance scalars and
        // EQUAL_NAN mirror the production dispatch (sfpu_operations.h ISCLOSE
        // branch: torch defaults rtol=1e-5, atol=1e-8, EQUAL_NAN=false).
        call_isclose_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, /*EQUAL_NAN=*/false, 8>(
            dst_in0, dst_in1, dst_out, VectorMode::RC, /*rtol_bits=*/0x3727c5acu, /*atol_bits=*/0x322bcc77u);
    }
    // Storm-lane S1 selectors (fresh_cpp/<op>.h semantic bodies).
    else if constexpr (
        FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::SUB &&
        static_cast<std::uint32_t>(formats.math) != static_cast<std::uint32_t>(DataFormat::Int32))
    {
        call_binary_float_sub_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::EQ &&
        static_cast<std::uint32_t>(formats.math) != static_cast<std::uint32_t>(DataFormat::Int32))
    {
        call_binary_comp_eq_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::FMOD)
    {
        call_binary_fmod_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::REMAINDER)
    {
        call_binary_remainder_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ATAN2)
    {
        call_atan2_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    // Lane DH fitted-kernel placeholder (tt-polynomial-fitter frontier atan
    // winner + the storm-S1 quadrant fixup; provenance header in
    // fresh_cpp/atan2_fitted.h): impl 2.
    else if constexpr (FRESH_CPP_IMPL == 2 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::ATAN2)
    {
        call_atan2_fitted_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::BITWISE_AND)
    {
        call_binary_bitwise_and_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    // Storm S2 (agent/storm-s2): canonical fresh_cpp/<op>.h semantic bodies.
    else if constexpr (
        FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::GCD &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Test-only fresh typed-C++ leg for the Int32 gcd production path
        // (metal ckernel_sfpu_gcd.h hand-issued REPLAY-loop kernel).
        call_gcd_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (
        FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::DIV_INT32_FLOOR &&
        static_cast<std::uint32_t>(formats.math) == static_cast<std::uint32_t>(DataFormat::Int32))
    {
        // Test-only fresh typed-C++ leg for the Int32 floor-division production
        // path (metal ckernel_sfpu_div_int32_floor.h).
        call_div_int32_floor_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    // Lane EU coverage expansion (binarypow-fresh row): fresh semantic pow
    // (impl 1) vs the byte-untouched calculate_sfpu_binary_pow hand kernel
    // (impl 3; the production POW dispatch routes through
    // calculate_sfpu_binary instead, so this kernel had zero nodes).
    else if constexpr (FRESH_CPP_IMPL == 1 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
    {
        call_binary_pow_fresh_cpp<DST_SYNC_MODE, is_fp32_dest_acc_en, 8>(dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else if constexpr (FRESH_CPP_IMPL == 3 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
    {
        SFPU_BINARY_CALL(
            DST_SYNC_MODE, is_fp32_dest_acc_en, calculate_sfpu_binary_pow, (APPROX_MODE, 8, is_fp32_dest_acc_en), dst_in0, dst_in1, dst_out, VectorMode::RC);
    }
    else
    {
        test_utils::call_binary_sfpu_operation<
            DST_SYNC_MODE,
            is_fp32_dest_acc_en,
            APPROX_MODE,
            SFPU_BINARY_OPERATION,
            ITERATIONS,
            formats.math>(dst_in0, dst_in1, dst_out);
    }
}

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

#ifndef SPEED_OF_LIGHT
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    const std::uint32_t num_faces   = params.num_faces;

    const std::uint32_t TILE_CNT = params.TILE_CNT;
#endif

    const DataCopyType data_copy_type = DataCopyType::A2D;

    {
        START_PERF_MEASURE("INIT")

        _llk_math_eltwise_unary_datacopy_init_<data_copy_type, is_fp32_dest_acc_en>(num_faces, formats.math);
        _llk_math_pack_sync_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
        _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

        test_utils::call_binary_sfpu_operation_init<APPROX_MODE, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, ITERATIONS>();
        // Lane EU coverage expansion: the impl-3 hand kernel's own init frame
        // (calculate_sfpu_binary_pow expects its 1/ln2 / -127 Prgm constants).
        if constexpr (FRESH_CPP_IMPL == 3 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
        {
            ckernel::sfpu::sfpu_binary_pow_init<APPROX_MODE>();
        }
        PROFILER_SYNC();
    }
    {
        START_PERF_MEASURE("TILE_LOOP")

        if constexpr (PERF_RUN_TYPE == PerfRunType::UNPACK_ISOLATE)
        {
            for (std::uint32_t loop = 0; loop < LOOP_FACTOR; ++loop)
            {
                for (std::uint32_t i = 0; i < TILE_CNT; ++i)
                {
                    // For unpack isolate scenario, math should only perform necessary synchronization and nothing else.
                    if constexpr (unpack_to_dest)
                    {
                        // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                        // "consumed" and can be overwritten with new data.
                        // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                        // this method will perform synchronization only and no actual data copy.
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            i % MAX_TILES_DEST, formats.math, formats.math);
                    }
                    else
                    {
                        // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                        _perf_math_loop_clear_valid<
                            /* src A */ true,
                            /* src B */ true>(
                            /* iterations*/ num_faces);
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
                    int block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (int block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        if constexpr (unpack_to_dest)
                        {
                            // In this case, unpacker needs software synchronization from math - to acknowledge that destination register is
                            // "consumed" and can be overwritten with new data.
                            // Due to the fact that BROADCAST_TYPE is always NONE in the test and combination of unpack_to_dest and 32b data is always set,
                            // this method will perform synchronization only and no actual data copy.
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }
                        else
                        {
                            // Perform only necessary hardware synchronization to indicate that source registers are consumed.
                            _perf_math_loop_clear_valid<
                                /* src A */ true,
                                /* src B */ true>(
                                /* iterations*/ num_faces);
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
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        // When data is not unpacked to dest, math needs to copy data from srcA to dest before starting SFPU operation.
                        // Otherwise, data is immediately ready in destination register.
                        if constexpr (!unpack_to_dest)
                        {
                            LLK_ASSERT(
                                (block_tile < get_dest_max_tiles<DstSync::SyncHalf, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                                "block_tile exceeds max dest tiles");
                            _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                                block_tile, formats.math, formats.math);
                        }

                        // The fresh semantic max/min contract requires physically adjacent
                        // inputs (in1 == in0 + 1) and in-place output; the previous modulo
                        // wrap produced (7, 0, 7) at the last dest tile and tripped the
                        // wrapper's device-side LLK_ASSERT (math core ebreak). Clamp the
                        // pair base instead: every iteration issues the same operation on a
                        // valid adjacent pair. The isolate scenarios measure timing, not
                        // payload placement, and the index is a runtime register for both
                        // impls, so neither instruction stream changes.
                        const std::uint32_t pair_base = std::min(block_tile, MAX_TILES_DEST - 2);
                        run_selected_binary_sfpu(pair_base, pair_base + 1, pair_base);
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
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_math_wait_for_dest_available_<DST_SYNC_MODE>();

                    // Copy from srcA to dest
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; ++block_tile)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_math_eltwise_unary_datacopy_<data_copy_type, DST_SYNC_MODE, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                            block_tile, formats.math, formats.math);

                        // Start SFPU binary operation
                        // The fresh semantic max/min contract requires physically adjacent
                        // inputs (in1 == in0 + 1) and in-place output; the previous modulo
                        // wrap produced (7, 0, 7) at the last dest tile and tripped the
                        // wrapper's device-side LLK_ASSERT (math core ebreak). Clamp the
                        // pair base instead: every iteration issues the same operation on a
                        // valid adjacent pair. The isolate scenarios measure timing, not
                        // payload placement, and the index is a runtime register for both
                        // impls, so neither instruction stream changes.
                        const std::uint32_t pair_base = std::min(block_tile, MAX_TILES_DEST - 2);
                        run_selected_binary_sfpu(pair_base, pair_base + 1, pair_base);
                    }

                    _llk_math_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }
        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_MATH

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
    const std::uint32_t num_faces   = params.num_faces;

    const std::uint32_t TILE_CNT = params.TILE_CNT;
#endif

    {
        START_PERF_MEASURE("INIT")

        // Configure packer hardware
        _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, FACE_R_DIM * FACE_C_DIM * num_faces);

        _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst, FACE_R_DIM, TILE_C_DIM, num_faces);
        // Initialize destination for packing
        _llk_pack_dest_init_<DST_SYNC_MODE, is_fp32_dest_acc_en>();

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
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
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
                    std::uint32_t block_tiles = std::min(TILE_CNT - block_start, MAX_TILES_DEST);

                    _llk_packer_wait_for_math_done_();
                    for (std::uint32_t block_tile = 0; block_tile < block_tiles; block_tile++)
                    {
                        LLK_ASSERT(
                            (block_tile < get_dest_max_tiles<DST_SYNC_MODE, is_fp32_dest_acc_en, DstTileShape::Tile32x32>()),
                            "block_tile exceeds max dest tiles");
                        _llk_pack_<DST_SYNC_MODE, is_fp32_dest_acc_en>(block_tile, PERF_ADDRESS(PERF_OUTPUT, block_start + block_tile));
                    }
                    _llk_pack_dest_section_done_<DST_SYNC_MODE, is_fp32_dest_acc_en>();
                }
            }
        }

        PROFILER_SYNC();
    }
}

#endif // LLK_TRISC_PACK
