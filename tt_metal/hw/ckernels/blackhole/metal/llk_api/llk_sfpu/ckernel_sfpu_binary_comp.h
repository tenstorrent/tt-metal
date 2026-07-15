// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <cstdint>

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu {

template <SfpuType Op>
inline constexpr bool is_fp32_equal_compare_v = Op == SfpuType::eq || Op == SfpuType::ne;

template <SfpuType Op>
inline constexpr bool is_fp32_strict_ordered_compare_v = Op == SfpuType::lt || Op == SfpuType::gt;

template <SfpuType Op>
inline constexpr bool is_fp32_weak_ordered_compare_v = Op == SfpuType::le || Op == SfpuType::ge;

template <SfpuType Op>
inline constexpr bool is_fp32_compare_v =
    is_fp32_equal_compare_v<Op> || is_fp32_strict_ordered_compare_v<Op> || is_fp32_weak_ordered_compare_v<Op>;

template <SfpuType>
inline constexpr bool unsupported_fp32_compare_v = false;

// fp32 total-order comparison helpers. Semantics preserved from the original raw-TTI
// implementation: |a|+|b| folds ±0/subnormals together (sum == 0) and detects NaN
// (as<vInt>(sum) > +inf bits). dst_reg[] uses sfpi row units (32/tile) vs the raw 64.
//
// NOTE: the fp32 comparison paths have no reliable tt-llk python coverage — the float
// comparison suite (test_sfpu_binary_float) disables Lt/Gt/Le/Ge/Eq/Ne because the
// generated stimuli produce near-ties. Validate at the ttnn level.
constexpr int FP32_INF_BITS = 0x7F800000;
constexpr std::uint32_t dst_tile_size_sfpi_comp = 32;

template <int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32_equal(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(is_fp32_equal_compare_v<RELATIONAL_OP>, "Supported operation types: eq, ne");

    constexpr float default_result = RELATIONAL_OP == SfpuType::eq ? 0.0f : 1.0f;
    constexpr float equal_result = RELATIONAL_OP == SfpuType::eq ? 1.0f : 0.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi_comp];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi_comp];
        sfpi::vFloat sum = sfpi::abs(a) + sfpi::abs(b);
        sfpi::vInt sum_bits = sfpi::as<sfpi::vInt>(sum);
        sfpi::vFloat result = default_result;

        // total-order a == b (a <= b && b <= a)
        v_if(a <= b && b <= a) { result = equal_result; }
        v_endif;
        // |a|+|b| == 0 treats all ±0/subnormals as equal
        v_if(sum == 0.0f) { result = equal_result; }
        v_endif;
        // NaN (|a|+|b| > inf) is never equal
        v_if(sum_bits > FP32_INF_BITS) { result = default_result; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_comp] = result;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32_strict_ordered(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(is_fp32_strict_ordered_compare_v<RELATIONAL_OP>, "Supported operation types: lt, gt");

    // gt(a,b) == lt(b,a): swap operands and compute the shared strict-less-than body.
    constexpr bool swap_operands = RELATIONAL_OP == SfpuType::gt;
    const std::uint32_t dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const std::uint32_t dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_a * dst_tile_size_sfpi_comp];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_b * dst_tile_size_sfpi_comp];
        sfpi::vFloat sum = sfpi::abs(a) + sfpi::abs(b);
        sfpi::vInt sum_bits = sfpi::as<sfpi::vInt>(sum);
        sfpi::vFloat result = 0.0f;

        // a < b, excluding both-±0/subnormal (sum == 0)
        v_if(a < b && sum != 0.0f) { result = 1.0f; }
        v_endif;
        // NaN (|a|+|b| > inf) is never strictly ordered
        v_if(sum_bits > FP32_INF_BITS) { result = 0.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_comp] = result;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32_weak_ordered(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(is_fp32_weak_ordered_compare_v<RELATIONAL_OP>, "Supported operation types: le, ge");

    // ge(a,b) == le(b,a): swap operands and compute the shared less-than-or-equal body.
    constexpr bool swap_operands = RELATIONAL_OP == SfpuType::ge;
    const std::uint32_t dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const std::uint32_t dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat a = sfpi::dst_reg[dst_index_a * dst_tile_size_sfpi_comp];
        sfpi::vFloat b = sfpi::dst_reg[dst_index_b * dst_tile_size_sfpi_comp];
        sfpi::vFloat sum = sfpi::abs(a) + sfpi::abs(b);
        sfpi::vInt sum_bits = sfpi::as<sfpi::vInt>(sum);
        sfpi::vFloat result = 1.0f;

        // a > b (excluding both-±0/subnormal) -> 0
        v_if(a > b && sum != 0.0f) { result = 0.0f; }
        v_endif;
        // NaN (|a|+|b| > inf) -> 0
        v_if(sum_bits > FP32_INF_BITS) { result = 0.0f; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi_comp] = result;
        sfpi::dst_reg++;
    }
}

// Float32 binary comparisons.
// - eq/ne: calculate_binary_comp_fp32_equal.
// - lt/gt: calculate_binary_comp_fp32_strict_ordered.
// - le/ge: calculate_binary_comp_fp32_weak_ordered.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    SfpuType RELATIONAL_OP,
    std::enable_if_t<is_fp32_compare_v<RELATIONAL_OP>, int> = 0>
inline void calculate_binary_comp_fp32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    if constexpr (is_fp32_equal_compare_v<RELATIONAL_OP>) {
        calculate_binary_comp_fp32_equal<ITERATIONS, RELATIONAL_OP>(dst_index_in0, dst_index_in1, dst_index_out);
    } else if constexpr (is_fp32_strict_ordered_compare_v<RELATIONAL_OP>) {
        calculate_binary_comp_fp32_strict_ordered<ITERATIONS, RELATIONAL_OP>(
            dst_index_in0, dst_index_in1, dst_index_out);
    } else if constexpr (is_fp32_weak_ordered_compare_v<RELATIONAL_OP>) {
        calculate_binary_comp_fp32_weak_ordered<ITERATIONS, RELATIONAL_OP>(dst_index_in0, dst_index_in1, dst_index_out);
    } else {
        static_assert(unsupported_fp32_compare_v<RELATIONAL_OP>, "Unsupported fp32 comparison operation");
    }
}

// Int32 relational comparisons. Normalize to LT(a, b) or GE(a, b):
//   lt(a,b) = LT(a,b)           gt(a,b) = LT(b,a)
//   ge(a,b) = GE(a,b)           le(a,b) = GE(b,a)
// Force b's top bit to 0 for LT or 1 for GE, subtract from a, then fold the
// original sign relationship with the subtraction result. The final shift
// converts the selected top bit to 0 or 1.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_int32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le ||
            RELATIONAL_OP == SfpuType::ge,
        "Supported operation types: lt, gt, le, ge");

    // INT32 load layout converts Dest sign-magnitude <-> 2's-complement, so the loaded
    // vInt lanes are true 2's-complement and sfpi's signed comparisons order them correctly.
    // dst_reg[] indexes in sfpi row units (32/tile), unlike the raw TT_SFPLOAD immediate (64).
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        sfpi::vInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        sfpi::vInt result = 0;

        if constexpr (RELATIONAL_OP == SfpuType::lt) {
            v_if(a < b) { result = 1; }
            v_endif;
        } else if constexpr (RELATIONAL_OP == SfpuType::gt) {
            v_if(a > b) { result = 1; }
            v_endif;
        } else if constexpr (RELATIONAL_OP == SfpuType::le) {
            v_if(a <= b) { result = 1; }
            v_endif;
        } else if constexpr (RELATIONAL_OP == SfpuType::ge) {
            v_if(a >= b) { result = 1; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>() = result;
        sfpi::dst_reg++;
    }
}

// UInt32/UInt16 relational comparisons. Normalize to LT(a, b) or GE(a, b):
//   lt(a,b) = LT(a,b)           gt(a,b) = LT(b,a)
//   ge(a,b) = GE(a,b)           le(a,b) = GE(b,a)
// UInt32 uses the same subtract/fold structure as Int32. On Blackhole, LO16
// zero-extends UInt16 values, so sign-magnitude compare matches uint16 order.
//
// NOTE (sfpi conversion): still raw TTI. The metal/tt-llk binary-comparison dispatch
// (sfpu_operations.h) only routes Int32 -> calculate_binary_comp_int32 and everything
// else -> calculate_binary_comp_fp32, so this uint ordering path (and calculate_binary_eq_int
// below) are ttnn-only and are neither instantiated nor testable by the tt-llk harness.
// The uint32 case additionally needs an MSB-fold to emulate unsigned ordering with signed
// sfpi compares; convert together with ttnn-level validation.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_comp_uint(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        DATA_FORMAT == DataFormat::UInt16 || DATA_FORMAT == DataFormat::UInt32,
        "Unsupported data format for calculate_binary_comp_uint(). Supported formats: UInt16, UInt32.");
    static_assert(
        RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le ||
            RELATIONAL_OP == SfpuType::ge,
        "Supported operation types: lt, gt, le, ge");

    constexpr bool use_ge = (RELATIONAL_OP == SfpuType::le || RELATIONAL_OP == SfpuType::ge);
    constexpr bool swap_operands = (RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le);
    constexpr bool needs_msb_handling = (DATA_FORMAT == DataFormat::UInt32);
    constexpr InstrModLoadStore ld_st_mod = needs_msb_handling ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr std::uint32_t a = p_sfpu::LREG0;
    constexpr std::uint32_t b = p_sfpu::LREG1;
    constexpr std::uint32_t scratch = p_sfpu::LREG2;
    constexpr std::uint32_t sign = use_ge ? 1 : 0;
    constexpr std::uint32_t result = use_ge ? a : b;
    constexpr std::uint32_t xor_src = use_ge ? b : a;
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const std::uint32_t dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(a, ld_st_mod, ADDR_MOD_7, dst_index_a * dst_tile_size);
        TT_SFPLOAD(b, ld_st_mod, ADDR_MOD_7, dst_index_b * dst_tile_size);

        if constexpr (needs_msb_handling) {
            TTI_SFPSETSGN(sign, b, scratch, 1);  // SFPSETSGN_MOD1_ARG_IMM
            TTI_SFPIADD(0, a, scratch, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPXOR(0, xor_src, result, 0);
            TTI_SFPOR(0, scratch, result, 0);
            TTI_SFPXOR(0, xor_src, result, 0);
            TTI_SFPSHFT((-31) & 0xfff, result, result, 1);  // SFPSHFT_MOD1_ARG_IMM
            TT_SFPSTORE(result, ld_st_mod, ADDR_MOD_6, dst_index_out * dst_tile_size);
        } else {
            if constexpr (use_ge) {
                TTI_SFPLE(0, a, b, 8);  // SFPLE_MOD1_SET_VD: b = (a >= b) ? -1 : 0
            } else {
                TTI_SFPGT(0, a, b, 8);  // SFPGT_MOD1_SET_VD: b = (a < b) ? -1 : 0
            }
            TTI_SFPSHFT((-31) & 0xfff, b, b, 1);  // SFPSHFT_MOD1_ARG_IMM
            TT_SFPSTORE(b, ld_st_mod, ADDR_MOD_6, dst_index_out * dst_tile_size);
        }
    }
}

// Integer equality comparisons: eq(a,b) and ne(a,b).
// XOR a and b; the result is zero if a == b. A conditional store writes
// the appropriate integer 0 or 1 result.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_eq_int(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt16 || DATA_FORMAT == DataFormat::UInt32,
        "Unsupported data format for calculate_binary_eq_int(). Supported formats: Int32, UInt16, UInt32.");
    static_assert(RELATIONAL_OP == SfpuType::eq || RELATIONAL_OP == SfpuType::ne, "Supported operation types: eq, ne");

    constexpr InstrModLoadStore ld_st_mod =
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    constexpr std::uint32_t a = p_sfpu::LREG0;
    constexpr std::uint32_t b = p_sfpu::LREG1;
    constexpr std::uint32_t one = p_sfpu::LREG2;
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr bool is_eq = (RELATIONAL_OP == SfpuType::eq);
    constexpr std::uint32_t default_result = is_eq ? p_sfpu::LCONST_0 : one;
    constexpr std::uint32_t equal_result = is_eq ? one : p_sfpu::LCONST_0;

    TTI_SFPLOADI(one, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(a, ld_st_mod, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(b, ld_st_mod, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        TT_SFPSTORE(default_result, ld_st_mod, ADDR_MOD_7, dst_index_out * dst_tile_size);

        TTI_SFPXOR(0, b, a, 0);
        TTI_SFPSETCC(0, a, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPSTORE(equal_result, ld_st_mod, ADDR_MOD_6, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}
}  //  namespace ckernel::sfpu
