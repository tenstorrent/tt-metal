// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
// This implements the "add 0x7fff + LSB" algorithm for correct tie-breaking
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    // Get the float32 bits as unsigned integer
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32)
    // This is needed for the tie-breaker: round to even
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - If lower 16 bits > 0x8000: overflow → rounds up
    // - If lower 16 bits < 0x8000: no overflow → rounds down
    // - If lower 16 bits = 0x8000 (tie) and lsb=0: 0x7fff+0=0xffff, no overflow → stays even
    // - If lower 16 bits = 0x8000 (tie) and lsb=1: 0x7fff+1=0x8000, overflow → rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32)
    bits = bits & 0xFFFF0000U;

    // Reinterpret back as float
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_power_21f_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // fmod(a, b) = a - trunc(a/b) * b
    //
    // Key insight: fmod result must satisfy: |result| < |b|
    // If this is violated, we need to correct the truncation.

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;
    sfpi::vFloat b_abs = sfpi::abs(b);

    // FIX 1: Handle a == b case (common for large values where a + offset = a)
    // When a == b, fmod(a, b) = 0
    // Use bit comparison: if a and b have same bits, result is 0
    sfpi::vFloat a_minus_b = a - b;

    // Step 1: Compute high-precision reciprocal 1/b
    sfpi::vFloat recip = ckernel::sfpu::_sfpu_reciprocal_<2>(b);

    // Step 2: Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * recip;

    // Step 3: Compute trunc(a/b) using hand-optimised trunc implementation
    sfpi::l_reg[sfpi::LRegs::LReg0] = div_result;
    _trunc_body_();
    sfpi::vFloat trunc_div = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat tmp2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat tmp3 = sfpi::l_reg[sfpi::LRegs::LReg3];

    // Step 4: Compute fmod = a - trunc(a/b) * b
    sfpi::vFloat result = a - trunc_div * b;

    // FIX 2: Post-correction - fmod result must satisfy |result| < |b|
    // If |result| >= |b|, the truncation was wrong by 1
    sfpi::vFloat result_abs = sfpi::abs(result);

    // If result >= b, we truncated too low, add/subtract b to correct
    v_if(result_abs >= b_abs) {
        // Determine correction direction based on sign of result
        v_if(result >= sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // result was positive and too big
        }
        v_else {
            result = result + b_abs;  // result was negative and too big (magnitude)
        }
        v_endif;
    }
    v_endif;

    // FIX 3: If a == b (within FP precision), result should be exactly 0
    // This handles edge case where a + small_offset = a due to FP precision
    v_if(a_minus_b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(0.0f); }
    v_endif;

    // FIX 4: Sign correction - fmod result must have same sign as 'a' (or be zero)
    // If a > 0 and result < 0, the truncation was 1 too high, need to add b
    // If a < 0 and result > 0, the truncation was 1 too low, need to subtract b
    // This fixes cases where a/b ≈ 0.9999999 but rounds to 1 due to reciprocal error
    v_if(a >= sfpi::vFloat(0.0f)) {
        // a is positive, result should be >= 0
        v_if(result < sfpi::vFloat(0.0f)) {
            result = result + b_abs;  // Correct: we over-truncated
        }
        v_endif;
    }
    v_else {
        // a is negative, result should be <= 0
        v_if(result > sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // Correct: we under-truncated
        }
        v_endif;
    }
    v_endif;

    // // Normal rounding
    // result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));

    // Kalai's rounding
    // result = float32_to_bf16_rne(result);

    return result;
}

/**
 * @brief Computes fmod(a, b) = a - trunc(a/b) * b for FP32 inputs
 *
 * This function implements the floating-point modulo operation using:
 * 1. High-precision reciprocal (1/b)
 * 2. Division a/b via multiplication by reciprocal
 * 3. Truncation towards zero using hand-optimised _trunc_body_()
 * 4. fmod = a - trunc(a/b) * b
 *
 * @param in0 The dividend (a)
 * @param in1 The divisor (b)
 * @return sfpi::vFloat Result of fmod(a, b)
 *
 * @note This is called when is_fp32_dest_acc_en == true
 */
// fmod implementation with edge case handling for large values
sfpi_inline sfpi::vFloat _sfpu_binary_power_61f_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // fmod(a, b) = a - trunc(a/b) * b
    //
    // Key insight: fmod result must satisfy: |result| < |b|
    // If this is violated, we need to correct the truncation.

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;
    sfpi::vFloat b_abs = sfpi::abs(b);

    // FIX 1: Handle a == b case (common for large values where a + offset = a)
    // When a == b, fmod(a, b) = 0
    // Use bit comparison: if a and b have same bits, result is 0
    sfpi::vFloat a_minus_b = a - b;

    // Step 1: Compute high-precision reciprocal 1/b
    sfpi::vFloat recip = ckernel::sfpu::_sfpu_reciprocal_<2>(b);

    // Step 2: Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * recip;

    // Step 3: Compute trunc(a/b) using hand-optimised trunc implementation
    sfpi::l_reg[sfpi::LRegs::LReg0] = div_result;
    _trunc_body_();
    sfpi::vFloat trunc_div = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat tmp2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat tmp3 = sfpi::l_reg[sfpi::LRegs::LReg3];

    // Step 4: Compute fmod = a - trunc(a/b) * b
    sfpi::vFloat result = a - trunc_div * b;

    // FIX 2: Post-correction - fmod result must satisfy |result| < |b|
    // If |result| >= |b|, the truncation was wrong by 1
    sfpi::vFloat result_abs = sfpi::abs(result);

    // If result >= b, we truncated too low, add/subtract b to correct
    v_if(result_abs >= b_abs) {
        // Determine correction direction based on sign of result
        v_if(result >= sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // result was positive and too big
        }
        v_else {
            result = result + b_abs;  // result was negative and too big (magnitude)
        }
        v_endif;
    }
    v_endif;

    // FIX 3: If a == b (within FP precision), result should be exactly 0
    // This handles edge case where a + small_offset = a due to FP precision
    v_if(a_minus_b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(0.0f); }
    v_endif;

    // FIX 4: Sign correction - fmod result must have same sign as 'a' (or be zero)
    // If a > 0 and result < 0, the truncation was 1 too high, need to add b
    // If a < 0 and result > 0, the truncation was 1 too low, need to subtract b
    // This fixes cases where a/b ≈ 0.9999999 but rounds to 1 due to reciprocal error
    v_if(a >= sfpi::vFloat(0.0f)) {
        // a is positive, result should be >= 0
        v_if(result < sfpi::vFloat(0.0f)) {
            result = result + b_abs;  // Correct: we over-truncated
        }
        v_endif;
    }
    v_else {
        // a is negative, result should be <= 0
        v_if(result > sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // Correct: we under-truncated
        }
        v_endif;
    }
    v_endif;

    return result;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_power_(sfpi::vFloat base, sfpi::vFloat pow);

// is_fp32_dest_acc_en == false
template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<false>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_21f_<false>(base, pow);
}

// is_fp32_dest_acc_en == true
template <>
sfpi_inline sfpi::vFloat _sfpu_binary_power_<true>(sfpi::vFloat base, sfpi::vFloat pow) {
    return _sfpu_binary_power_61f_(base, pow);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_pow(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_power_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void sfpu_binary_pow_init() {
    // sfpi::vConstFloatPrgm0 = 1.442695f;
    // sfpi::vConstFloatPrgm1 = -127.0f;
    // sfpi::vConstFloatPrgm2 = std::numeric_limits<float>::quiet_NaN();
    _init_sfpu_reciprocal_<false>();
}

}  // namespace sfpu
}  // namespace ckernel
