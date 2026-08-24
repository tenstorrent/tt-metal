// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "llk_defs.h"

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
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::DEFAULT_TENSOR_SHAPE, formats.unpack_A_src, formats.unpack_A_dst);
    for (std::uint32_t i = 0; i < params.TILE_CNT; i++)
    {
        _llk_unpack_A_<BROADCAST_TYPE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[i]), formats.unpack_A_src, formats.unpack_A_dst);
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
    const bool is_int_fpu_en         = false;
    constexpr DataCopyType copy_type = (BROADCAST_TYPE == BroadcastType::NONE || unpack_to_dest) ? DataCopyType::A2D : DataCopyType::B2D;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
        _llk_math_eltwise_unary_datacopy_init_wrapper_<copy_type, is_fp32_dest_acc_en, BROADCAST_TYPE, is_int_fpu_en, PackMode::Default>(
            4 /* num_faces */, formats.math);
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            _llk_math_eltwise_unary_datacopy_<copy_type, DstSync::SyncHalf, is_fp32_dest_acc_en, BROADCAST_TYPE, unpack_to_dest>(
                tile, formats.math, formats.math);
        }
        _llk_math_eltwise_unary_datacopy_uninit_<BROADCAST_TYPE, unpack_to_dest>();

        test_utils::call_binary_sfpu_operation_init<APPROX_MODE, is_fp32_dest_acc_en, SFPU_BINARY_OPERATION, 32 /* iterations */, formats.math>();

        // Lane EU coverage expansion: impl 3 = the byte-untouched
        // calculate_sfpu_binary_pow kernel (metal ckernel_sfpu_binary_pow.h,
        // corpus id metal__ckernel_sfpu_binary_pow — zero standalone nodes
        // before this selector: the production POW dispatch above routes
        // through calculate_sfpu_binary instead).  Its own init frame programs
        // the 1/ln2 / -127 Prgm constants the kernel core expects.
        if constexpr (FRESH_CPP_IMPL == 3 && SFPU_BINARY_OPERATION == ckernel::BinaryOp::POW)
        {
            ckernel::sfpu::sfpu_binary_pow_init<APPROX_MODE>();
        }

        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; tile += 2)
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
                test_utils::call_binary_sfpu_operation<
                    DstSync::SyncHalf,
                    is_fp32_dest_acc_en,
                    APPROX_MODE,
                    SFPU_BINARY_OPERATION,
                    32 /* iterations */,
                    formats.math>(tile, tile + 1, tile);
            }
        }
        _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(formats.pack_src, formats.pack_dst, 16 * 16 /* tile_size */);

    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);

    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    for (int block = 0; block < params.NUM_BLOCKS; ++block)
    {
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < params.NUM_TILES_IN_BLOCK; ++tile)
        {
            const std::uint32_t result_tile = block * params.NUM_TILES_IN_BLOCK + tile;
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(tile, L1_ADDRESS(params.buffer_Res[result_tile]));
        }
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    }
}

#endif
