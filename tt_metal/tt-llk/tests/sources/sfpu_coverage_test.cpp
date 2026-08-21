// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Lane EU LLK-coverage-expansion vehicle: races the corpus-uncovered SFPU
// kernels (manifest classes D-ABSENT — zero dispatch anywhere under tests/
// before this file) against fresh semantic bodies, without touching any
// production tree (LLK-pristine rule R7: all wiring is test-side).
//
// COVERAGE_OP selects the raced kernel pair (values mirror
// test_sfpu_coverage.py::CoverageOp):
//   1 rotate90       metal ckernel_sfpu_alt_complex_rotate90.h
//   2 unarybitwise   metal ckernel_sfpu_bitwise.h        (COVERAGE_SUBOP: 0 AND / 1 OR / 2 XOR)
//   3 addrsqrt       metal experimental ckernel_sfpu_add_rsqrt.h
//   4 smoothstep     metal experimental ckernel_sfpu_smoothstep.h
//   5 tiledprod      metal ckernel_sfpu_tiled_prod.h
//   6 zeropad        legacy experimental ckernel_sfpu_zero_pad.h
//   7 sparsekfilter  legacy experimental ckernel_sfpu_sparse_k_filter.h
//   8 customadd      metal experimental ckernel_sfpu_custom_add.h   (two-tile)
//   9 copydest       metal ckernel_sfpu_copy_dest_values.h          (two-tile)
//  10 intsum         metal ckernel_sfpu_int_sum.h        (COVERAGE_SUBOP: 0 COL / 1 ROW)
// FRESH_CPP_IMPL: 0 = byte-untouched production kernel (hand arm),
//                 1 = fresh_cpp semantic body (sem arm).

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_debug.h"
#include "llk_defs.h"
// params.h FIRST (build.h carries the COVERAGE_OP/COVERAGE_SUBOP/
// FRESH_CPP_IMPL template defines; it must precede the #ifndef defaults and
// the derived constants below — the perf twin's include order).
#include "params.h"

// Globals
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#ifndef COVERAGE_OP
#define COVERAGE_OP 1
#endif
#ifndef COVERAGE_SUBOP
#define COVERAGE_SUBOP 0
#endif
#ifndef FRESH_CPP_IMPL
#define FRESH_CPP_IMPL 0
#endif

// Two-tile ops unpack buffer_B into Dst tile 1 next to buffer_A's tile 0.
static constexpr bool COVERAGE_TWO_TILE = (COVERAGE_OP == 8 || COVERAGE_OP == 9);
// copydest writes tile 1 (out-of-place); everything else lands in tile 0.
static constexpr std::uint32_t COVERAGE_PACK_TILE = (COVERAGE_OP == 9) ? 1 : 0;

// Fixed sparse-k-filter field geometry, mirrored by the python golden.
static constexpr std::uint32_t SKF_BANK_MASK         = 0x3F;
static constexpr std::uint32_t SKF_MY_BANK           = 5;
static constexpr std::uint32_t SKF_GLOBAL_BANK_SHIFT = 10;
static constexpr std::uint32_t SKF_WITHIN_BANK_MASK  = 0x3FF;
static constexpr std::uint32_t SKF_OUT_SHIFT         = 0;

// Fixed zero-pad row split (VALID rows kept, the rest scrubbed), mirrored by
// the python golden.
static constexpr int ZERO_PAD_VALID_ROWS = 24;
static constexpr int ZERO_PAD_TOTAL_ROWS = 32;

// Fixed smoothstep edges (edge1 - edge0 = 1 so inv_delta = 1), mirrored by
// the python golden.  The kernel contract takes edge0/edge1/inv_delta floats.
static constexpr float SMOOTHSTEP_EDGE0                  = -0.5f;
static constexpr float SMOOTHSTEP_EDGE1                  = 0.5f;
static constexpr float SMOOTHSTEP_INV_DELTA              = 1.0f;
static constexpr std::uint32_t SMOOTHSTEP_EDGE0_BITS     = 0xBF000000u; // -0.5f
static constexpr std::uint32_t SMOOTHSTEP_INV_DELTA_BITS = 0x3F800000u; // 1.0f

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_A.h"
#include "llk_unpack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t UNPACK_FMT = UNPACK_A_IN;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, UNPACK_FMT, FACE_R_DIM, FACE_R_DIM, 4 /* num_faces */, 4 /* num_faces */);
    _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
        0 /* transpose_of_faces */, 0 /* within_face_16x16_transpose */, ckernel::make_tensor_shape_from_legacy(FACE_R_DIM, 4), UNPACK_FMT, UNPACK_FMT);
    _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_A[0]), UNPACK_FMT, UNPACK_FMT);
    if constexpr (COVERAGE_TWO_TILE)
    {
        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(L1_ADDRESS(params.buffer_B[0]), UNPACK_FMT, UNPACK_FMT);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "ckernel_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "params.h"

using namespace ckernel;

// SFPU call/init macro layer FIRST: its init header forward-declares the
// production init functions, and -Wredundant-decls requires declarations to
// precede the kernel headers' definitions (own include block so clang-format
// keeps the order).
#include "llk_sfpu/llk_math_eltwise_binary_sfpu_macros.h"
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
// The legacy Converter helper several production kernels expect in scope
// (the sfpu_operations.h include-order convention).
#include "sfpu/ckernel_sfpu_converter.h"
// Byte-untouched production kernels under race (hand arms).
#include "experimental/llk_sfpu/ckernel_sfpu_add_rsqrt.h"
#include "experimental/llk_sfpu/ckernel_sfpu_custom_add.h"
#include "experimental/llk_sfpu/ckernel_sfpu_smoothstep.h"
#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_copy_dest_values.h"
#include "llk_sfpu/ckernel_sfpu_int_sum.h"
#include "llk_sfpu/ckernel_sfpu_tiled_prod.h"
#include "sfpu/experimental/ckernel_sfpu_sparse_k_filter.h"
#include "sfpu/experimental/ckernel_sfpu_zero_pad.h"
// Fresh semantic bodies (sem arms; storm contract, fresh_cpp/README.md).
#include "fresh_cpp/addrsqrt.h"
#include "fresh_cpp/copydest.h"
#include "fresh_cpp/customadd.h"
#include "fresh_cpp/intsum.h"
#include "fresh_cpp/rotate90.h"
#include "fresh_cpp/smoothstep.h"
#include "fresh_cpp/sparsekfilter.h"
#include "fresh_cpp/tiledprod.h"
#include "fresh_cpp/unarybitwise.h"
#include "fresh_cpp/zeropad.h"

void run_kernel(RUNTIME_PARAMETERS)
{
    const bool is_int_fpu_en    = false;
    const std::uint8_t MATH_FMT = UNPACK_A_IN;

    _llk_math_pack_sync_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    _llk_math_hw_configure_<is_fp32_dest_acc_en>(MATH_FMT, MATH_FMT);
    _llk_math_eltwise_unary_datacopy_init_wrapper_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, is_int_fpu_en, PackMode::Default>(
        4 /* num_faces */, MATH_FMT);
    _llk_math_wait_for_dest_available_<DstSync::SyncHalf>();
    _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
        0 /* dst_index */, MATH_FMT, MATH_FMT);
    if constexpr (COVERAGE_TWO_TILE)
    {
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            1 /* dst_index */, MATH_FMT, MATH_FMT);
    }
    // Reset dest addressing before the SFPU op so the dest RWC is at the
    // tile-0 base (the binop-scalar/ternary SFPU test convention).
    _llk_math_eltwise_unary_datacopy_uninit_<BroadcastType::NONE, unpack_to_dest>();

    SFPU_UNARY_INIT(unused);

#if FRESH_CPP_IMPL == 0
    // ---- hand arms: byte-untouched production kernels + their own inits ----
#if COVERAGE_OP == 1
    sfpu::alt_complex_rotate90_init();
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_alt_complex_rotate90, (APPROX_MODE, 4), 0, VectorMode::RC);
#elif COVERAGE_OP == 2
#if COVERAGE_SUBOP == 0
    sfpu::bitwise_and_init();
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        calculate_sfpu_unary_bitwise,
        (APPROX_MODE, sfpu::UnaryBitwiseOp::AND, DataFormat::Int32, 8),
        0,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#elif COVERAGE_SUBOP == 1
    sfpu::bitwise_or_init();
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        calculate_sfpu_unary_bitwise,
        (APPROX_MODE, sfpu::UnaryBitwiseOp::OR, DataFormat::Int32, 8),
        0,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#else
    sfpu::bitwise_xor_init();
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        calculate_sfpu_unary_bitwise,
        (APPROX_MODE, sfpu::UnaryBitwiseOp::XOR, DataFormat::Int32, 8),
        0,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#endif
#elif COVERAGE_OP == 3
    sfpu::init_add_rsqrt<APPROX_MODE>();
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        calculate_add_rsqrt,
        (APPROX_MODE, 8, is_fp32_dest_acc_en, false /* FAST_APPROX */),
        0,
        VectorMode::RC,
        SFPU_UNARY_SCALAR);
#elif COVERAGE_OP == 4
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(
        DstSync::SyncHalf, is_fp32_dest_acc_en, smoothstep_tile_face, 0, VectorMode::RC, SMOOTHSTEP_EDGE0, SMOOTHSTEP_EDGE1, SMOOTHSTEP_INV_DELTA);
#elif COVERAGE_OP == 5
    sfpu::tiled_prod_init();
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_tiled_prod, (APPROX_MODE, 8), 0, VectorMode::None);
#elif COVERAGE_OP == 6
    SFPU_UNARY_CALL(
        DstSync::SyncHalf, is_fp32_dest_acc_en, _zero_pad_tile_, (is_fp32_dest_acc_en, ZERO_PAD_VALID_ROWS, ZERO_PAD_TOTAL_ROWS), 0, VectorMode::None);
#elif COVERAGE_OP == 7
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        _sparse_k_filter_tile_,
        (32 /* ITERATIONS */, SKF_BANK_MASK, SKF_MY_BANK, SKF_GLOBAL_BANK_SHIFT, SKF_WITHIN_BANK_MASK, SKF_OUT_SHIFT),
        0,
        VectorMode::None);
#elif COVERAGE_OP == 8
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DstSync::SyncHalf, is_fp32_dest_acc_en, my_add_tile_face, 0, VectorMode::RC, 0 /* in0 */, 1 /* in1 */, 0 /* out */);
#elif COVERAGE_OP == 9
    sfpu::copy_dest_value_init();
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        copy_dest_value,
        (DataFormat::Float16_b, APPROX_MODE, 8),
        0,
        VectorMode::RC,
        0 /* in */,
        1 /* out */,
        0 /* unused */);
#elif COVERAGE_OP == 10
    sfpu::sum_int_init<APPROX_MODE>();
#if COVERAGE_SUBOP == 0
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_sum_int_col, (APPROX_MODE), 0, VectorMode::None);
#else
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_sum_int_row, (APPROX_MODE), 0, VectorMode::None);
#endif
#else
#error "unknown COVERAGE_OP"
#endif

#else // FRESH_CPP_IMPL == 1
    // ---- sem arms: fresh semantic bodies (full-tile, VectorMode::None) ----
#if COVERAGE_OP == 1
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_rotate90_fresh_cpp, (16), 0, VectorMode::None);
#elif COVERAGE_OP == 2
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_unary_bitwise_fresh_cpp, (COVERAGE_SUBOP, 32), 0, VectorMode::None, SFPU_UNARY_SCALAR);
#elif COVERAGE_OP == 3
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_add_rsqrt_fresh_cpp, (32), 0, VectorMode::None, SFPU_UNARY_SCALAR);
#elif COVERAGE_OP == 4
    SFPU_UNARY_CALL(
        DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_smoothstep_fresh_cpp, (32), 0, VectorMode::None, SMOOTHSTEP_EDGE0_BITS, SMOOTHSTEP_INV_DELTA_BITS);
#elif COVERAGE_OP == 5
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_tiled_prod_fresh_cpp, (9), 0, VectorMode::None);
#elif COVERAGE_OP == 6
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_zero_pad_fresh_cpp, (ZERO_PAD_VALID_ROWS, ZERO_PAD_TOTAL_ROWS), 0, VectorMode::None);
#elif COVERAGE_OP == 7
    SFPU_UNARY_CALL(
        DstSync::SyncHalf,
        is_fp32_dest_acc_en,
        calculate_sparse_k_filter_fresh_cpp,
        (SKF_BANK_MASK, SKF_MY_BANK, SKF_GLOBAL_BANK_SHIFT, SKF_WITHIN_BANK_MASK, SKF_OUT_SHIFT, 32),
        0,
        VectorMode::None);
#elif COVERAGE_OP == 8
    sfpu::call_custom_add_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(0, 1, 0, VectorMode::RC);
#elif COVERAGE_OP == 9
    sfpu::call_copy_dest_fresh_cpp<DstSync::SyncHalf, is_fp32_dest_acc_en, 8>(0, 1, VectorMode::RC);
#elif COVERAGE_OP == 10
    SFPU_UNARY_CALL(DstSync::SyncHalf, is_fp32_dest_acc_en, calculate_int_sum_fresh_cpp, (COVERAGE_SUBOP), 0, VectorMode::None);
#else
#error "unknown COVERAGE_OP"
#endif
#endif // FRESH_CPP_IMPL

    _llk_math_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint8_t PACK_FMT = UNPACK_A_IN;

    _llk_pack_hw_configure_wrapper_<is_fp32_dest_acc_en, PackMode::Default>(PACK_FMT, PACK_FMT, 16 * 16 * 4 /* tile_size */);
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(PACK_FMT);
    _llk_pack_dest_init_wrapper_<DstSync::SyncHalf, is_fp32_dest_acc_en, PackMode::Default>();

    _llk_packer_wait_for_math_done_();
    _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en, ckernel::PackMode::Default>(COVERAGE_PACK_TILE, L1_ADDRESS(params.buffer_Res[0]));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
}

#endif
