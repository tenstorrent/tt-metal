// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include <cstdint>

#include "ckernel_addrmod.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_comp.h"

namespace ckernel::sfpu {

template <CompareOp Op>
inline constexpr bool is_fp32_equal_compare_v = Op == CompareOp::eq || Op == CompareOp::ne;

template <CompareOp Op>
inline constexpr bool is_fp32_strict_ordered_compare_v = Op == CompareOp::lt || Op == CompareOp::gt;

template <CompareOp Op>
inline constexpr bool is_fp32_weak_ordered_compare_v = Op == CompareOp::le || Op == CompareOp::ge;

template <CompareOp Op>
inline constexpr bool is_fp32_compare_v =
    is_fp32_equal_compare_v<Op> || is_fp32_strict_ordered_compare_v<Op> || is_fp32_weak_ordered_compare_v<Op>;

template <CompareOp>
inline constexpr bool unsupported_fp32_compare_v = false;

template <int ITERATIONS, CompareOp RELATIONAL_OP>
inline void calculate_binary_comp_fp32_equal(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(is_fp32_equal_compare_v<RELATIONAL_OP>, "Supported operation types: eq, ne");

    constexpr std::uint32_t a = p_sfpu::LREG0;
    constexpr std::uint32_t b = p_sfpu::LREG1;
    constexpr std::uint32_t abs_a = p_sfpu::LREG2;
    constexpr std::uint32_t abs_b = p_sfpu::LREG3;
    constexpr std::uint32_t sum = p_sfpu::LREG4;
    constexpr std::uint32_t inf = p_sfpu::LREG5;
    constexpr std::uint32_t default_result = RELATIONAL_OP == CompareOp::eq ? p_sfpu::LCONST_0 : p_sfpu::LCONST_1;
    constexpr std::uint32_t equal_result = RELATIONAL_OP == CompareOp::eq ? p_sfpu::LCONST_1 : p_sfpu::LCONST_0;
    constexpr std::uint32_t dst_tile_size = 64;

    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(a, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TT_SFPSTORE(default_result, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, b, abs_b, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, a, abs_a, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, abs_a, abs_b, sum, 0);
        TTI_SFPXOR(0, b, a, 0);

        // if abs(a) + abs(b) == 0; this allows us to treat all ±subnormals as equal
        TTI_SFPSETCC(0, sum, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);

        // if abs(a) + abs(b) <= inf; rejects NaN
        TTI_SFPIADD(0, inf, sum, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
        // if a ^ b == 0; requires both values to be bitwise identical
        TTI_SFPSETCC(0, a, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_2, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <int ITERATIONS, CompareOp RELATIONAL_OP>
inline void calculate_binary_comp_fp32_strict_ordered(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(is_fp32_strict_ordered_compare_v<RELATIONAL_OP>, "Supported operation types: lt, gt");

    constexpr std::uint32_t a = p_sfpu::LREG0;
    constexpr std::uint32_t b = p_sfpu::LREG1;
    constexpr std::uint32_t abs_a = p_sfpu::LREG2;
    constexpr std::uint32_t abs_b = p_sfpu::LREG3;
    constexpr std::uint32_t sum = p_sfpu::LREG4;
    constexpr std::uint32_t copy = p_sfpu::LREG5;
    constexpr std::uint32_t inf = p_sfpu::LREG6;
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr bool swap_operands = RELATIONAL_OP == CompareOp::gt;
    const std::uint32_t dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const std::uint32_t dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(a, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_b * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, a, abs_a, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, b, abs_b, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, abs_a, abs_b, sum, 0);
        TTI_SFPMOV(0, a, copy, 0);

        // if abs(a) + abs(b) != 0; rejects if both are ±subnormal
        TTI_SFPSETCC(0, sum, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        // if abs(a) + abs(b) <= inf; rejects NaN
        TTI_SFPIADD(0, inf, sum, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);

        // reject if a >= b
        TTI_SFPSWAP(0, a, b, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPXOR(0, a, copy, 0);
        TTI_SFPSETCC(0, copy, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TT_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <int ITERATIONS, CompareOp RELATIONAL_OP>
inline void calculate_binary_comp_fp32_weak_ordered(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(is_fp32_weak_ordered_compare_v<RELATIONAL_OP>, "Supported operation types: le, ge");

    constexpr std::uint32_t a = p_sfpu::LREG0;
    constexpr std::uint32_t b = p_sfpu::LREG1;
    constexpr std::uint32_t abs_a = p_sfpu::LREG2;
    constexpr std::uint32_t abs_b = p_sfpu::LREG3;
    constexpr std::uint32_t sum = p_sfpu::LREG4;
    constexpr std::uint32_t copy = p_sfpu::LREG5;
    constexpr std::uint32_t inf = p_sfpu::LREG6;
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr bool swap_operands = RELATIONAL_OP == CompareOp::le;
    const std::uint32_t dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const std::uint32_t dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

    TTI_SFPLOADI(inf, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(a, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_b * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, a, abs_a, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, b, abs_b, 1);  // SFPSETSGN_MOD1_ARG_IMM

        TTI_SFPMAD(p_sfpu::LCONST_1, abs_a, abs_b, sum, 0);
        TTI_SFPMOV(0, a, copy, 0);

        // if abs(a) + abs(b) != 0; rejects if both are ±subnormal
        TTI_SFPSETCC(0, sum, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);

        // reject if original comparison is false
        TTI_SFPSWAP(0, a, b, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPXOR(0, a, copy, 0);
        TTI_SFPSETCC(0, copy, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);

        // if abs(a) + abs(b) > inf; a or b is NaN
        TTI_SFPIADD(0, inf, sum, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

// Float32 binary comparisons.
// - eq/ne: calculate_binary_comp_fp32_equal.
// - lt/gt: calculate_binary_comp_fp32_strict_ordered.
// - le/ge: calculate_binary_comp_fp32_weak_ordered.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    CompareOp RELATIONAL_OP,
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
template <bool APPROXIMATION_MODE, int ITERATIONS, CompareOp RELATIONAL_OP>
inline void calculate_binary_comp_int32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        RELATIONAL_OP == CompareOp::lt || RELATIONAL_OP == CompareOp::gt || RELATIONAL_OP == CompareOp::le ||
            RELATIONAL_OP == CompareOp::ge,
        "Supported operation types: lt, gt, le, ge");

    constexpr bool use_ge = (RELATIONAL_OP == CompareOp::le || RELATIONAL_OP == CompareOp::ge);
    constexpr bool swap_operands = (RELATIONAL_OP == CompareOp::gt || RELATIONAL_OP == CompareOp::le);
    constexpr std::uint32_t a = p_sfpu::LREG0;
    constexpr std::uint32_t b = p_sfpu::LREG1;
    constexpr std::uint32_t scratch = p_sfpu::LREG2;
    constexpr std::uint32_t sign = use_ge ? 1 : 0;
    constexpr std::uint32_t tmp = use_ge ? b : a;
    constexpr std::uint32_t xor_src = use_ge ? a : b;
    constexpr std::uint32_t dst_tile_size = 64;

    const std::uint32_t dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const std::uint32_t dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(a, InstrModLoadStore::INT32, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(b, InstrModLoadStore::INT32, ADDR_MOD_3, dst_index_b * dst_tile_size);

        TTI_SFPSETSGN(sign, b, scratch, 1);  // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPIADD(0, a, scratch, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPXOR(0, xor_src, tmp, 0);
        TTI_SFPOR(0, scratch, tmp, 0);
        TTI_SFPXOR(0, b, a, 0);
        TTI_SFPSHFT((-31) & 0xfff, a, a, 1);  // SFPSHFT_MOD1_ARG_IMM
        TT_SFPSTORE(a, InstrModLoadStore::INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}

// UInt32/UInt16 relational comparisons. Normalize to LT(a, b) or GE(a, b):
//   lt(a,b) = LT(a,b)           gt(a,b) = LT(b,a)
//   ge(a,b) = GE(a,b)           le(a,b) = GE(b,a)
// UInt32 uses the same subtract/fold structure as Int32. For UInt16, a - b
// cannot overflow int32; the sign bit gives LT, and SFPNOT turns LT into GE.
template <bool APPROXIMATION_MODE, int ITERATIONS, CompareOp RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_comp_uint(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        DATA_FORMAT == DataFormat::UInt16 || DATA_FORMAT == DataFormat::UInt32,
        "Unsupported data format for calculate_binary_comp_uint(). Supported formats: UInt16, UInt32.");
    static_assert(
        RELATIONAL_OP == CompareOp::lt || RELATIONAL_OP == CompareOp::gt || RELATIONAL_OP == CompareOp::le ||
            RELATIONAL_OP == CompareOp::ge,
        "Supported operation types: lt, gt, le, ge");

    constexpr bool use_ge = (RELATIONAL_OP == CompareOp::le || RELATIONAL_OP == CompareOp::ge);
    constexpr bool swap_operands = (RELATIONAL_OP == CompareOp::gt || RELATIONAL_OP == CompareOp::le);
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
        TT_SFPLOAD(a, ld_st_mod, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(b, ld_st_mod, ADDR_MOD_3, dst_index_b * dst_tile_size);

        if constexpr (needs_msb_handling) {
            TTI_SFPSETSGN(sign, b, scratch, 1);  // SFPSETSGN_MOD1_ARG_IMM
            TTI_SFPIADD(0, a, scratch, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPXOR(0, xor_src, result, 0);
            TTI_SFPOR(0, scratch, result, 0);
            TTI_SFPXOR(0, xor_src, result, 0);
            TTI_SFPSHFT((-31) & 0xfff, result, result, 1);  // SFPSHFT_MOD1_ARG_IMM
            TT_SFPSTORE(result, ld_st_mod, ADDR_MOD_2, dst_index_out * dst_tile_size);
        } else {
            // Signed subtraction cannot overflow for UInt16; the sign bit gives the strict comparison.
            TTI_SFPIADD(0, a, b, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
            if constexpr (use_ge) {
                TTI_SFPNOT(0, b, b, 0);
            }
            TTI_SFPSHFT((-31) & 0xfff, b, b, 1);  // SFPSHFT_MOD1_ARG_IMM
            TT_SFPSTORE(b, ld_st_mod, ADDR_MOD_2, dst_index_out * dst_tile_size);
        }
    }
}

// Integer equality comparisons: eq(a,b) and ne(a,b).
// XOR a and b; the result is zero if a == b. A conditional store writes
// the appropriate integer 0 or 1 result.
template <bool APPROXIMATION_MODE, int ITERATIONS, CompareOp RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_eq_int(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt16 || DATA_FORMAT == DataFormat::UInt32,
        "Unsupported data format for calculate_binary_eq_int(). Supported formats: Int32, UInt16, UInt32.");
    static_assert(
        RELATIONAL_OP == CompareOp::eq || RELATIONAL_OP == CompareOp::ne, "Supported operation types: eq, ne");

    constexpr InstrModLoadStore ld_st_mod =
        (DATA_FORMAT == DataFormat::UInt16) ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;
    constexpr std::uint32_t a = p_sfpu::LREG0;
    constexpr std::uint32_t b = p_sfpu::LREG1;
    constexpr std::uint32_t one = p_sfpu::LREG2;
    constexpr std::uint32_t dst_tile_size = 64;

    constexpr bool is_eq = (RELATIONAL_OP == CompareOp::eq);
    constexpr std::uint32_t default_result = is_eq ? p_sfpu::LCONST_0 : one;
    constexpr std::uint32_t equal_result = is_eq ? one : p_sfpu::LCONST_0;

    TTI_SFPLOADI(one, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(a, ld_st_mod, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(b, ld_st_mod, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TT_SFPSTORE(default_result, ld_st_mod, ADDR_MOD_3, dst_index_out * dst_tile_size);

        TTI_SFPXOR(0, b, a, 0);
        TTI_SFPSETCC(0, a, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPSTORE(equal_result, ld_st_mod, ADDR_MOD_2, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

// Per-op init for the binary compare kernels (lt/gt/le/ge/eq/ne and their _int variants), which
// until now got ADDR_MOD_6 only from the SfpuType-selected LLK init.
inline void binary_comp_init() {
    // The compare kernels store through ADDR_MOD_2, which addr_mod_base maps to hardware slot 6.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

// SfpuType-selected entry points, kept for existing callers. They forward to the CompareOp versions above.

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    calculate_binary_comp_fp32<APPROXIMATION_MODE, ITERATIONS, _sfpu_type_to_compare_op_<RELATIONAL_OP>()>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_int32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    calculate_binary_comp_int32<APPROXIMATION_MODE, ITERATIONS, _sfpu_type_to_compare_op_<RELATIONAL_OP>()>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_comp_uint(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    calculate_binary_comp_uint<APPROXIMATION_MODE, ITERATIONS, _sfpu_type_to_compare_op_<RELATIONAL_OP>(), DATA_FORMAT>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_eq_int(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    calculate_binary_eq_int<APPROXIMATION_MODE, ITERATIONS, _sfpu_type_to_compare_op_<RELATIONAL_OP>(), DATA_FORMAT>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

// Op class for elementwise compare of two float tiles in Dest: out = (in0 OP in1) ? 1.0 : 0.0.
template <bool APPROXIMATION_MODE, CompareOp RELATIONAL_OP, int ITERATIONS = 8>
struct BinaryComp : SfpuBinaryOp<BinaryComp<APPROXIMATION_MODE, RELATIONAL_OP, ITERATIONS>> {
    static constexpr auto& calculate = calculate_binary_comp_fp32<APPROXIMATION_MODE, ITERATIONS, RELATIONAL_OP>;
    static inline __attribute__((always_inline)) void init_op() { binary_comp_init(); }
};

// Op class for elementwise compare of two Int32, UInt32 or UInt16 tiles in Dest: out = (in0 OP in1) ? 1 : 0.
// eq/ne use calculate_binary_eq_int; lt/gt/le/ge use calculate_binary_comp_int32 (Int32) or
// calculate_binary_comp_uint (UInt32/UInt16).
template <bool APPROXIMATION_MODE, CompareOp RELATIONAL_OP, DataFormat DATA_FORMAT, int ITERATIONS = 8>
struct BinaryCompInt : SfpuBinaryOp<BinaryCompInt<APPROXIMATION_MODE, RELATIONAL_OP, DATA_FORMAT, ITERATIONS>> {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    static inline __attribute__((always_inline)) void calculate(
        const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
        if constexpr (RELATIONAL_OP == CompareOp::eq || RELATIONAL_OP == CompareOp::ne) {
            calculate_binary_eq_int<APPROXIMATION_MODE, ITERATIONS, RELATIONAL_OP, DATA_FORMAT>(
                dst_index_in0, dst_index_in1, dst_index_out);
        } else if constexpr (DATA_FORMAT == DataFormat::Int32) {
            calculate_binary_comp_int32<APPROXIMATION_MODE, ITERATIONS, RELATIONAL_OP>(
                dst_index_in0, dst_index_in1, dst_index_out);
        } else {
            calculate_binary_comp_uint<APPROXIMATION_MODE, ITERATIONS, RELATIONAL_OP, DATA_FORMAT>(
                dst_index_in0, dst_index_in1, dst_index_out);
        }
    }
    static inline __attribute__((always_inline)) void init_op() { binary_comp_init(); }
};

}  //  namespace ckernel::sfpu
