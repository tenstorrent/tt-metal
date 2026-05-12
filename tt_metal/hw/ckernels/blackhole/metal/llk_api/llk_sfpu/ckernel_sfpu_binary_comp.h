// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Int32 binary comparison for relational ops (signed).
// All ops reduce to computing LT(X, Y) with optional operand swap and result inversion:
//   lt(A,B) = LT(A,B)           gt(A,B) = LT(B,A)
//   ge(A,B) = NOT LT(A,B)       le(A,B) = NOT LT(B,A)
// When sign bits of X and Y match, signed subtraction can't overflow and the sign of (X-Y)
// is the answer. When sign bits differ, X < Y iff X is negative (MSB(X) == 1).
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(
        RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le ||
            RELATIONAL_OP == SfpuType::ge,
        "Supported operation types: lt, gt, le, ge");

    // GT and LE swap operands (compute sign(B-A)); LT and GE use natural order (compute sign(A-B)).
    // LE and GE then invert the result (LE = NOT GT, GE = NOT LT).
    constexpr bool swap_operands = (RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le);
    constexpr bool invert_result = (RELATIONAL_OP == SfpuType::le || RELATIONAL_OP == SfpuType::ge);
    constexpr uint dst_tile_size = 64;

    // Loop-invariant invert mask; hoisted out of the unrolled loop to avoid re-issuing
    // TTI_SFPLOADI on every iteration.
    if constexpr (invert_result) {
        TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0x01);
    }

    // Pick X/Y order once; LREG0 holds X, LREG1 holds Y.
    const uint dst_index_x = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_y = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_7, dst_index_x * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_7, dst_index_y * dst_tile_size);

        // Extract sign bits of X (LREG0) and Y (LREG1)
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        // LREG3 -> 0 for inputs of same sign, 1 for inputs of different signs
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

        // if (LREG3 == 0) -> same signs: signed subtract + extract sign of X-Y
        TTI_SFPSETCC(0, p_sfpu::LREG3, 0 /*unused*/, SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // else -> different signs: load 0, then load 1 if MSB(X) != 0 (X is negative, so X < Y)
        TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        TTI_SFPENCC(0, 0, 0, 0);

        if constexpr (invert_result) {
            TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);
        }

        TT_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_6, dst_index_out * dst_tile_size);
    }
}

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

// uint32 & uint16binary comparison for relational ops
// All ops reduce to computing LT(X, Y) with optional operand swap and result inversion:
//   lt(A,B) = LT(A,B)           gt(A,B) = LT(B,A)
//   ge(A,B) = NOT LT(A,B)       le(A,B) = NOT LT(B,A)
// When MSBs of X and Y match, signed subtraction gives the correct unsigned comparison.
// When MSBs differ, X < Y iff MSB(X) == 0.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP, DataFormat DATA_FORMAT>
inline void calculate_binary_comp_uint(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(
        DATA_FORMAT == DataFormat::UInt16 || DATA_FORMAT == DataFormat::UInt32,
        "Unsupported data format for calculate_binary_comp_uint(). Supported formats: UInt16, UInt32.");
    static_assert(
        RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le ||
            RELATIONAL_OP == SfpuType::ge,
        "Supported operation types: lt, gt, le, ge");

    // GT and LE swap operands (compute sign(B-A)); LT and GE use natural order (compute sign(A-B)).
    // LE and GE then invert the result (LE = NOT GT, GE = NOT LT).
    constexpr bool swap_operands = (RELATIONAL_OP == SfpuType::gt || RELATIONAL_OP == SfpuType::le);
    constexpr bool invert_result = (RELATIONAL_OP == SfpuType::le || RELATIONAL_OP == SfpuType::ge);
    // UInt32 needs full 32-bit loads + MSB disambiguation; UInt16 fits in the low half so we use
    // a 16-bit load (zero-extended into the LREG) and skip the MSB-handling path.
    constexpr bool needs_msb_handling = (DATA_FORMAT == DataFormat::UInt32);
    constexpr std::uint32_t LD_ST_MOD = needs_msb_handling ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
    constexpr uint dst_tile_size = 64;

    // Loop-invariant invert mask; hoisted out of the unrolled loop to avoid re-issuing
    // TTI_SFPLOADI on every iteration.
    if constexpr (invert_result) {
        TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0x01);
    }

    // Pick X/Y order once; LREG0 holds X, LREG1 holds Y.
    const uint dst_index_x = swap_operands ? dst_index_in1 : dst_index_in0;
    const uint dst_index_y = swap_operands ? dst_index_in0 : dst_index_in1;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, LD_ST_MOD, ADDR_MOD_7, dst_index_x * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, LD_ST_MOD, ADDR_MOD_7, dst_index_y * dst_tile_size);

        if constexpr (needs_msb_handling) {
            // Extract MSBs of X (LREG0) and Y (LREG1)
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

            // LREG3 -> 0 for inputs with same MSB, 1 for inputs with different MSBs
            TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

            // if (LREG3 == 0) -> same MSBs: signed subtract + extract sign
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0 /*unused*/, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
            // else -> different MSBs: load 0, then load 1 if MSB(X) == 0 (X is smaller)
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // LREG1 = 1 XOR MSB(X)
            TTI_SFPENCC(0, 0, 0, 0);
        } else {
            // 16-bit case: signed subtract can't overflow, sign bit gives the answer directly.
            // LREG1 = LREG0 - LREG1 (imod=6: dst = src - dst)
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            // Extract sign bit: 1 if negative, 0 otherwise
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        }

        if constexpr (invert_result) {
            TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);
        }

        TT_SFPSTORE(p_sfpu::LREG1, LD_ST_MOD, ADDR_MOD_6, dst_index_out * dst_tile_size);
    }
}
}  //  namespace ckernel::sfpu
