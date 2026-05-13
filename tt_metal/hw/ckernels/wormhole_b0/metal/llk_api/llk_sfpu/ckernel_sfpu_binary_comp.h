// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

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
inline constexpr bool is_fp32_compare_v = is_fp32_equal_compare_v<Op> || is_fp32_strict_ordered_compare_v<Op> || is_fp32_weak_ordered_compare_v<Op>;

template <SfpuType>
inline constexpr bool unsupported_fp32_compare_v = false;

template <int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32_equal(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(is_fp32_equal_compare_v<RELATIONAL_OP>, "Supported operation types: eq, ne");

    constexpr uint A = p_sfpu::LREG0;
    constexpr uint B = p_sfpu::LREG1;
    constexpr uint ABS_A = p_sfpu::LREG2;
    constexpr uint ABS_B = p_sfpu::LREG3;
    constexpr uint SUM = p_sfpu::LREG4;
    constexpr uint INF = p_sfpu::LREG5;
    constexpr uint default_result = RELATIONAL_OP == SfpuType::eq ? p_sfpu::LCONST_0 : p_sfpu::LCONST_1;
    constexpr uint equal_result = RELATIONAL_OP == SfpuType::eq ? p_sfpu::LCONST_1 : p_sfpu::LCONST_0;
    constexpr uint dst_tile_size = 64;

    TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(B, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        TT_SFPSTORE(default_result, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, B, ABS_B, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, A, ABS_A, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, ABS_A, ABS_B, SUM, 0);
        TTI_SFPXOR(0, B, A, 0);

        // if abs(a) + abs(b) == 0; this allows us to treat all ±subnormals as equal
        TTI_SFPSETCC(0, SUM, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);

        // if abs(a) + abs(b) <= inf; rejects NaN
        TTI_SFPIADD(0, INF, SUM, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
        // if a ^ b == 0; requires both values to be bitwise identical
        TTI_SFPSETCC(0, A, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_2, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32_strict_ordered(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(is_fp32_strict_ordered_compare_v<RELATIONAL_OP>, "Supported operation types: lt, gt");

    constexpr uint A = p_sfpu::LREG0;
    constexpr uint B = p_sfpu::LREG1;
    constexpr uint ABS_A = p_sfpu::LREG2;
    constexpr uint ABS_B = p_sfpu::LREG3;
    constexpr uint SUM = p_sfpu::LREG4;
    constexpr uint COPY = p_sfpu::LREG5;
    constexpr uint INF = p_sfpu::LREG6;
    constexpr uint dst_tile_size = 64;

    constexpr bool swap_operands = RELATIONAL_OP == SfpuType::gt;
    const uint dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

    TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_b * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, A, ABS_A, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, B, ABS_B, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, ABS_A, ABS_B, SUM, 0);
        TTI_SFPMOV(0, A, COPY, 0);

        // if abs(a) + abs(b) != 0; rejects if both are ±subnormal
        TTI_SFPSETCC(0, SUM, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        // if abs(a) + abs(b) <= inf; rejects NaN
        TTI_SFPIADD(0, INF, SUM, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);

        // reject if a >= b
        TTI_SFPSWAP(0, A, B, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPXOR(0, A, COPY, 0);
        TTI_SFPSETCC(0, COPY, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TT_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_2, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);
    }
}

template <int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_fp32_weak_ordered(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(is_fp32_weak_ordered_compare_v<RELATIONAL_OP>, "Supported operation types: le, ge");

    constexpr uint A = p_sfpu::LREG0;
    constexpr uint B = p_sfpu::LREG1;
    constexpr uint ABS_A = p_sfpu::LREG2;
    constexpr uint ABS_B = p_sfpu::LREG3;
    constexpr uint SUM = p_sfpu::LREG4;
    constexpr uint COPY = p_sfpu::LREG5;
    constexpr uint INF = p_sfpu::LREG6;
    constexpr uint dst_tile_size = 64;

    constexpr bool swap_operands = RELATIONAL_OP == SfpuType::le;
    const uint dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

    TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_b * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, A, ABS_A, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, B, ABS_B, 1); // SFPSETSGN_MOD1_ARG_IMM

        TTI_SFPMAD(p_sfpu::LCONST_1, ABS_A, ABS_B, SUM, 0);
        TTI_SFPMOV(0, A, COPY, 0);

        // if abs(a) + abs(b) != 0; rejects if both are ±subnormal
        TTI_SFPSETCC(0, SUM, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);

        // reject if original comparison is false
        TTI_SFPSWAP(0, A, B, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        TTI_SFPXOR(0, A, COPY, 0);
        TTI_SFPSETCC(0, COPY, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, dst_index_out * dst_tile_size);
        TTI_SFPENCC(0, 0, 0, 0);

        // if abs(a) + abs(b) > inf; a or b is NaN
        TTI_SFPIADD(0, INF, SUM, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
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
    SfpuType RELATIONAL_OP,
    std::enable_if_t<is_fp32_compare_v<RELATIONAL_OP>, int> = 0>
inline void calculate_binary_comp_fp32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    if constexpr (is_fp32_equal_compare_v<RELATIONAL_OP>) {
        calculate_binary_comp_fp32_equal<ITERATIONS, RELATIONAL_OP>(dst_index_in0, dst_index_in1, dst_index_out);
    } else if constexpr (is_fp32_strict_ordered_compare_v<RELATIONAL_OP>) {
        calculate_binary_comp_fp32_strict_ordered<ITERATIONS, RELATIONAL_OP>(
            dst_index_in0, dst_index_in1, dst_index_out);
    } else if constexpr (is_fp32_weak_ordered_compare_v<RELATIONAL_OP>) {
        calculate_binary_comp_fp32_weak_ordered<ITERATIONS, RELATIONAL_OP>(
            dst_index_in0, dst_index_in1, dst_index_out);
    } else {
        static_assert(unsupported_fp32_compare_v<RELATIONAL_OP>, "Unsupported fp32 comparison operation");
    }
}

// Int32 binary comparisons.
// Compute LT(X, Y) or GE(X, Y) directly; gt/le swap operands first:
//   lt(A,B) = LT(A,B)           gt(A,B) = LT(B,A)
//   ge(A,B) = GE(A,B)           le(A,B) = GE(B,A)
// SFPSETSGN builds the adjusted subtrahend without a constant load:
//   LT: D = B with sign bit cleared, then D = A - D
//   GE: D = B with sign bit set,     then D = A - D
// The xor/or/xor fold combines sign(A), sign(B), and sign(D); the final shift extracts
// the boolean result as 0 or 1.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(
        RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le ||
            RELATIONAL_OP == SfpuType::ge,
        "Supported operation types: lt, gt, le, ge");

    constexpr bool use_ge = (RELATIONAL_OP == SfpuType::le || RELATIONAL_OP == SfpuType::ge);
    constexpr bool swap_operands = (RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le);
    constexpr uint A = p_sfpu::LREG0;
    constexpr uint B = p_sfpu::LREG1;
    constexpr uint D = p_sfpu::LREG2;
    constexpr uint SIGN = use_ge ? 1 : 0;
    constexpr uint TMP = use_ge ? B : A;
    constexpr uint XOR_SRC = use_ge ? A : B;
    constexpr uint dst_tile_size = 64;

    const uint dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A, INT32, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, INT32, ADDR_MOD_3, dst_index_b * dst_tile_size);

        TTI_SFPSETSGN(SIGN, B, D, 1);
        TTI_SFPIADD(0, A, D, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPXOR(0, XOR_SRC, TMP, 0);
        TTI_SFPOR(0, D, TMP, 0);
        TTI_SFPXOR(0, B, A, 0);
        TTI_SFPSHFT((-31) & 0xfff, A, A, 1);
        TT_SFPSTORE(A, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}

// UInt32 and UInt16 binary comparisons.
// UInt32 computes LT(X, Y) or GE(X, Y) directly; gt/le swap operands first:
//   lt(A,B) = LT(A,B)           gt(A,B) = LT(B,A)
//   ge(A,B) = GE(A,B)           le(A,B) = GE(B,A)
// Like int32, UInt32 uses SFPSETSGN to form B with either a cleared or set MSB before
// subtracting, then folds the original MSB relationship with the subtract result.
// UInt16 stays simpler: A - B cannot overflow int32, so its sign bit gives LT directly
// and SFPNOT turns LT into GE when needed.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_comp_uint(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
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
    constexpr std::uint32_t LD_ST_MOD = needs_msb_handling ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr uint A = p_sfpu::LREG0;
    constexpr uint B = p_sfpu::LREG1;
    constexpr uint D = p_sfpu::LREG2;
    constexpr uint SIGN = use_ge ? 1 : 0;
    constexpr uint RESULT = use_ge ? A : B;
    constexpr uint XOR_SRC = use_ge ? B : A;
    constexpr uint dst_tile_size = 64;

    const uint dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A, LD_ST_MOD, ADDR_MOD_3, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, LD_ST_MOD, ADDR_MOD_3, dst_index_b * dst_tile_size);

        if constexpr (needs_msb_handling) {
            TTI_SFPSETSGN(SIGN, B, D, 1);
            TTI_SFPIADD(0, A, D, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPXOR(0, XOR_SRC, RESULT, 0);
            TTI_SFPOR(0, D, RESULT, 0);
            TTI_SFPXOR(0, XOR_SRC, RESULT, 0);
            TTI_SFPSHFT((-31) & 0xfff, RESULT, RESULT, 1);
            TT_SFPSTORE(RESULT, LD_ST_MOD, ADDR_MOD_2, dst_index_out * dst_tile_size);
        } else {
            // Signed subtraction cannot overflow for UInt16; the sign bit gives the strict comparison.
            TTI_SFPIADD(0, A, B, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
            if constexpr (use_ge) {
                TTI_SFPNOT(0, B, B, 0);
            }
            TTI_SFPSHFT((-31) & 0xfff, B, B, 1);
            TT_SFPSTORE(B, LD_ST_MOD, ADDR_MOD_2, dst_index_out * dst_tile_size);
        }
    }
}
}  //  namespace ckernel::sfpu
