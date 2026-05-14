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
        TT_SFPLOAD(A, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_in0 * dst_tile_size);
        TT_SFPLOAD(B, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_in1 * dst_tile_size);
        TT_SFPSTORE(default_result, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, B, ABS_B, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, A, ABS_A, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPMAD(p_sfpu::LCONST_1, ABS_A, ABS_B, SUM, 0);

        TTI_SFPLE(0, B, A, 1); // SFPLE_MOD1_SET_CC
        // if total-order a == b
        TTI_SFPLE(0, A, B, 1); // SFPLE_MOD1_SET_CC
        // if abs(a) + abs(b) <= inf; rejects NaN
        TTI_SFPIADD(0, INF, SUM, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
        TT_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_out * dst_tile_size);

        TTI_SFPENCC(0, 0, 0, 0);

        // if abs(a) + abs(b) == 0; this allows us to treat all ±subnormals as equal
        TTI_SFPSETCC(0, SUM, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TT_SFPSTORE(equal_result, InstrModLoadStore::DEFAULT, ADDR_MOD_6, dst_index_out * dst_tile_size);

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
    constexpr uint INF = p_sfpu::LREG5;
    constexpr uint dst_tile_size = 64;

    constexpr bool swap_operands = RELATIONAL_OP == SfpuType::gt;
    const uint dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

    TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_b * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, A, ABS_A, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, B, ABS_B, 1); // SFPSETSGN_MOD1_ARG_IMM

        TTI_SFPMAD(p_sfpu::LCONST_1, ABS_A, ABS_B, SUM, 0);
        // if total-order a < b
        TTI_SFPGT(0, A, B, 1); // SFPGT_MOD1_SET_CC

        // if abs(a) + abs(b) != 0; rejects if both are ±subnormal
        TTI_SFPSETCC(0, SUM, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        // if abs(a) + abs(b) <= inf; rejects NaN
        TTI_SFPIADD(0, INF, SUM, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
        TT_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_6, dst_index_out * dst_tile_size);

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
    constexpr uint INF = p_sfpu::LREG5;
    constexpr uint dst_tile_size = 64;

    constexpr bool swap_operands = RELATIONAL_OP == SfpuType::ge;
    const uint dst_index_a = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_b = swap_operands ? dst_index_in0 : dst_index_in1;

    TTI_SFPLOADI(INF, sfpi::SFPLOADI_MOD0_FLOATB, 0x7f80);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(A, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_b * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LCONST_1, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_out * dst_tile_size);

        TTI_SFPSETSGN(0, A, ABS_A, 1); // SFPSETSGN_MOD1_ARG_IMM
        TTI_SFPSETSGN(0, B, ABS_B, 1); // SFPSETSGN_MOD1_ARG_IMM

        TTI_SFPMAD(p_sfpu::LCONST_1, ABS_A, ABS_B, SUM, 0);
        // if total-order a > b
        TTI_SFPGT(0, B, A, 1); // SFPGT_MOD1_SET_CC

        // if abs(a) + abs(b) != 0; rejects if both are ±subnormal
        TTI_SFPSETCC(0, SUM, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, dst_index_out * dst_tile_size);

        TTI_SFPENCC(0, 0, 0, 0);

        // if abs(a) + abs(b) > inf; a or b is NaN
        TTI_SFPIADD(0, INF, SUM, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
        TT_SFPSTORE(p_sfpu::LCONST_0, InstrModLoadStore::DEFAULT, ADDR_MOD_6, dst_index_out * dst_tile_size);

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

// Int32 relational comparisons. Normalize to LT(A, B) or GE(A, B):
//   lt(A,B) = LT(A,B)           gt(A,B) = LT(B,A)
//   ge(A,B) = GE(A,B)           le(A,B) = GE(B,A)
// Force B's top bit to 0 for LT or 1 for GE, subtract from A, then fold the
// original sign relationship with the subtraction result. The final shift
// converts the selected top bit to 0 or 1.
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
        TT_SFPLOAD(A, INT32, ADDR_MOD_7, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, INT32, ADDR_MOD_7, dst_index_b * dst_tile_size);

        TTI_SFPSETSGN(SIGN, B, D, 1);
        TTI_SFPIADD(0, A, D, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPXOR(0, XOR_SRC, TMP, 0);
        TTI_SFPOR(0, D, TMP, 0);
        TTI_SFPXOR(0, B, A, 0);
        TTI_SFPSHFT((-31) & 0xfff, A, A, 1);
        TT_SFPSTORE(A, INT32, ADDR_MOD_6, dst_index_out * dst_tile_size);
    }
}

// UInt32/UInt16 relational comparisons. Normalize to LT(A, B) or GE(A, B):
//   lt(A,B) = LT(A,B)           gt(A,B) = LT(B,A)
//   ge(A,B) = GE(A,B)           le(A,B) = GE(B,A)
// UInt32 uses the same subtract/fold structure as Int32. On Blackhole, LO16
// zero-extends UInt16 values, so sign-magnitude compare matches uint16 order.
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
        TT_SFPLOAD(A, LD_ST_MOD, ADDR_MOD_7, dst_index_a * dst_tile_size);
        TT_SFPLOAD(B, LD_ST_MOD, ADDR_MOD_7, dst_index_b * dst_tile_size);

        if constexpr (needs_msb_handling) {
            TTI_SFPSETSGN(SIGN, B, D, 1);
            TTI_SFPIADD(0, A, D, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPXOR(0, XOR_SRC, RESULT, 0);
            TTI_SFPOR(0, D, RESULT, 0);
            TTI_SFPXOR(0, XOR_SRC, RESULT, 0);
            TTI_SFPSHFT((-31) & 0xfff, RESULT, RESULT, 1);
            TT_SFPSTORE(RESULT, LD_ST_MOD, ADDR_MOD_6, dst_index_out * dst_tile_size);
        } else {
            if constexpr (use_ge) {
                TTI_SFPLE(0, A, B, 8); // SFPLE_MOD1_SET_VD: B = (A >= B) ? -1 : 0
            } else {
                TTI_SFPGT(0, A, B, 8); // SFPGT_MOD1_SET_VD: B = (A < B) ? -1 : 0
            }
            TTI_SFPSHFT((-31) & 0xfff, B, B, 1);
            TT_SFPSTORE(B, LD_ST_MOD, ADDR_MOD_6, dst_index_out * dst_tile_size);
        }
    }
}
}  //  namespace ckernel::sfpu
