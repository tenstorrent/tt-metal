// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "ckernel_addrmod.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Comparison ops use int32 subtract (whose result is also in the int32 range) + sign check.
// In order to avoid overflow for inputs of opposite signs, the output is determined directly from a sign check.
template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr uint dst_tile_size = 64;
        // operand A
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        // operand B
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_in1 * dst_tile_size);

        // Extract sign bits of A and B
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);

        // LREG_3 -> 0 for inputs of same sign, 1 for inputs of different signs
        TTI_SFPXOR(0, p_sfpu::LREG2, p_sfpu::LREG3, 0);

        if constexpr (RELATIONAL_OP == SfpuType::lt) {
            // if (LREG_3 == 0) -> use int32 subtract + extract sign
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0 /*unused*/, SFPSETCC_MOD1_LREG_EQ0);
            // (A - B) -> Use 6 or LO16 as imod to convert operand B to 2's complement
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
            // else -> load 0 for inputs of opposite signs
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x00);
            // Load 1 if input A is negative
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);

            // LREG_1 -> dest
            TT_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);

        } else if constexpr (RELATIONAL_OP == SfpuType::gt) {
            // if (LREG_3 == 0) -> use int32 subtract + extract sign
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0, SFPSETCC_MOD1_LREG_EQ0);
            // (B - A) -> Use 6 or LO16 as imod to convert operand B to 2's complement
            TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            // else -> load 0 for inputs of opposite signs
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x00);
            // Load 1 if input A is non-negative
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);

            // LREG_0 -> dest
            TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);

        } else if constexpr (RELATIONAL_OP == SfpuType::ge) {
            // Implements GE by using LT logic and then inverting the result
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0 /*unused*/, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x00);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_NE0);
            TTI_SFPLOADI(p_sfpu::LREG1, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);

            // XOR with 1 to invert the result
            TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);

            // LREG_1 -> dest
            TT_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);

        } else if constexpr (RELATIONAL_OP == SfpuType::le) {
            // Implements LE by using GT logic and then inverting the result
            TTI_SFPSETCC(0, p_sfpu::LREG3, 0, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 6);
            TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 1);
            TTI_SFPCOMPC(0 /*unused*/, 0 /*unused*/, 0 /*unused*/, 0 /*unused*/);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x00);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPLOADI(p_sfpu::LREG0, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPENCC(0, 0, 0, 0);

            // XOR with 1 to invert the result
            TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0x01);
            TTI_SFPXOR(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);

            // LREG_0 -> dest
            TT_SFPSTORE(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_out * dst_tile_size);
        }

        sfpi::dst_reg++;
    }
}

// IEEE-754 float32 NaN: exponent all 1s and nonzero mantissa => |bits| > 0x7F800000.
sfpi_inline auto is_nan(sfpi::vInt abs_bits) { return abs_bits > 0x7F800000; }
// IEEE-754 float32 infinity: exponent all 1s, mantissa zero => |bits| == 0x7F800000.
sfpi_inline auto is_inf(sfpi::vInt abs_bits) { return abs_bits == 0x7F800000; }

template <SfpuType Op>
inline constexpr bool is_fp32_equal_compare_v = Op == SfpuType::eq || Op == SfpuType::ne;

template <SfpuType Op>
inline constexpr bool is_fp32_ordered_compare_v =
    Op == SfpuType::lt || Op == SfpuType::gt || Op == SfpuType::le || Op == SfpuType::ge;

template <SfpuType Op>
inline constexpr bool is_fp32_compare_v = is_fp32_equal_compare_v<Op> || is_fp32_ordered_compare_v<Op>;

template <SfpuType RELATIONAL_OP, std::enable_if_t<is_fp32_equal_compare_v<RELATIONAL_OP>, int> = 0>
sfpi_inline sfpi::vFloat binary_comp_fp32_equal_mask(sfpi::vFloat in0, sfpi::vFloat in1) {
    constexpr bool is_eq = (RELATIONAL_OP == SfpuType::eq);
    const sfpi::vFloat positive = is_eq ? sfpi::vConst1 : sfpi::vConst0;
    const sfpi::vFloat negative = is_eq ? sfpi::vConst0 : sfpi::vConst1;

    sfpi::vFloat mask = negative;
    sfpi::vInt in0_bits = sfpi::reinterpret<sfpi::vInt>(in0);
    sfpi::vInt in1_bits = sfpi::reinterpret<sfpi::vInt>(in1);
    sfpi::vInt in0_abs = in0_bits & 0x7FFFFFFF;
    sfpi::vInt in1_abs = in1_bits & 0x7FFFFFFF;

    // Equality (±0 / identical +inf/-inf bit patterns) then strip NaN lanes. For ne, NaN strip must
    // follow so unordered NaN lanes stay "not equal" (1) and are not confused with same-infinity eq.
    v_if(
        (in0 == in1) ||                                 // Handle equal values
        (in0_abs == 0 && in1_abs == 0) ||               // Handle ±0.0
        ((in0_bits == in1_bits) && is_inf(in0_abs))) {  // Handle +inf/-inf bit patterns
        mask = positive;
    }
    v_endif;
    v_if(is_nan(in0_abs) || is_nan(in1_abs)) { mask = negative; }
    v_endif;
    return mask;
}

template <SfpuType RELATIONAL_OP, std::enable_if_t<is_fp32_ordered_compare_v<RELATIONAL_OP>, int> = 0>
sfpi_inline sfpi::vFloat binary_comp_fp32_ordered_mask(sfpi::vFloat in0, sfpi::vFloat in1) {
    sfpi::vInt in0_bits = sfpi::reinterpret<sfpi::vInt>(in0);
    sfpi::vInt in1_bits = sfpi::reinterpret<sfpi::vInt>(in1);
    sfpi::vInt in0_abs = in0_bits & 0x7FFFFFFF;
    sfpi::vInt in1_abs = in1_bits & 0x7FFFFFFF;
    sfpi::vFloat result = sfpi::vConst0;

    // +inf/+inf and -inf/-inf: IEEE tie rules.

    if constexpr (RELATIONAL_OP == SfpuType::lt) {
        v_if(in0 < in1) { result = sfpi::vConst1; }
        v_endif;
        v_if((in0_bits == in1_bits) && is_inf(in0_abs)) { result = sfpi::vConst0; }
        v_endif;
    } else if constexpr (RELATIONAL_OP == SfpuType::gt) {
        v_if(in0 > in1) { result = sfpi::vConst1; }
        v_endif;
        v_if((in0_bits == in1_bits) && is_inf(in0_abs)) { result = sfpi::vConst0; }
        v_endif;
    } else if constexpr (RELATIONAL_OP == SfpuType::le) {
        v_if(in0 <= in1) { result = sfpi::vConst1; }
        v_endif;
        v_if((in0_bits == in1_bits) && is_inf(in0_abs)) { result = sfpi::vConst1; }
        v_endif;
    } else {
        v_if(in0 >= in1) { result = sfpi::vConst1; }
        v_endif;
        v_if((in0_bits == in1_bits) && is_inf(in0_abs)) { result = 1.0f; }
        v_endif;
    }

    // ±0.0 vs ±0.0: IEEE ties; HW may order signed zeros differently.
    if constexpr (RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt) {
        v_if((in0_abs == 0) && (in1_abs == 0)) { result = sfpi::vConst0; }
        v_endif;
    } else {
        v_if((in0_abs == 0) && (in1_abs == 0)) { result = sfpi::vConst1; }
        v_endif;
    }

    v_if(is_nan(in0_abs) || is_nan(in1_abs)) { result = sfpi::vConst0; }
    v_endif;
    return result;
}

// Float32 binary comparisons.
// - lt/gt/le/ge: binary_comp_fp32_ordered_mask.
// - eq/ne: binary_comp_fp32_equal_mask.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    SfpuType RELATIONAL_OP,
    std::enable_if_t<is_fp32_compare_v<RELATIONAL_OP>, int> = 0>
inline void calculate_binary_comp_float(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = sfpi::vConst0;

        if constexpr (is_fp32_equal_compare_v<RELATIONAL_OP>) {
            result = binary_comp_fp32_equal_mask<RELATIONAL_OP>(in0, in1);
        } else {
            result = binary_comp_fp32_ordered_mask<RELATIONAL_OP>(in0, in1);
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, SfpuType RELATIONAL_OP>
inline void calculate_binary_comp_uint16(
    const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {
    static_assert(RELATIONAL_OP == SfpuType::lt || RELATIONAL_OP == SfpuType::gt, "Supported operation types: lt, gt");
    constexpr uint32_t dst_tile_size = 64;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        if constexpr (RELATIONAL_OP == SfpuType::lt) {
            // Load operand A as uint16 for lt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
            // Load operand B as uint16 for lt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
        } else if constexpr (RELATIONAL_OP == SfpuType::gt) {
            // Load operand A as uint16 for gt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG0, LO16, ADDR_MOD_3, dst_index_in1 * dst_tile_size);
            // Load operand B as uint16 for gt operation (zero-extended to 32 bits)
            TT_SFPLOAD(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_index_in0 * dst_tile_size);
        }
        // LREG1 = LREG0 - LREG1 = A - B (imod=6 does dst = src - dst)
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
        // Extract sign bit: logical right shift by 31 -> 1 if negative (A < B), 0 otherwise
        TTI_SFPSHFT((-31) & 0xfff, p_sfpu::LREG1, p_sfpu::LREG1, 1);
        // Store the result in the destination register
        TT_SFPSTORE(p_sfpu::LREG1, LO16, ADDR_MOD_3, dst_index_out * dst_tile_size);
        sfpi::dst_reg++;
    }
}
}  //  namespace ckernel::sfpu
