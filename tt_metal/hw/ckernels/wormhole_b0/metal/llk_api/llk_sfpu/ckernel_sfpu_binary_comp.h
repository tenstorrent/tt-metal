// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
        TT_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, dst_index_x * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_index_y * dst_tile_size);

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

        TT_SFPSTORE(p_sfpu::LREG1, INT32, ADDR_MOD_2, dst_index_out * dst_tile_size);
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
inline void calculate_binary_comp_fp32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
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
        TT_SFPLOAD(p_sfpu::LREG0, LD_ST_MOD, ADDR_MOD_3, dst_index_x * dst_tile_size);
        TT_SFPLOAD(p_sfpu::LREG1, LD_ST_MOD, ADDR_MOD_3, dst_index_y * dst_tile_size);

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

        TT_SFPSTORE(p_sfpu::LREG1, LD_ST_MOD, ADDR_MOD_2, dst_index_out * dst_tile_size);
    }
}
}  //  namespace ckernel::sfpu
