// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_remainder_int32.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// FMOD = a - trunc(a / b) * b
// Implemented using 32-bit integer remainder kernel (see ckernel_sfpu_remainder_int32.h)
sfpi_inline void calculate_fmod_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Load signed inputs
    // Equivalent to: sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi] = a_signed;
    sfpi::vInt a_signed = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());
    // Equivalent to: sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi] = b_signed;
    sfpi::vInt b_signed = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].get());

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // FMOD sign handling (result has the same sign as a)
    v_if(a_signed < 0) { r = -r; }
    v_endif;

    // Equivalent to: sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
    __builtin_rvtt_sfpstore(
        r.get(), 4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].get());
}

template <bool is_fp32_dest_acc_en = false>
sfpi_inline sfpi::vFloat _sfpu_binary_fmod_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // fmod(a, b) = a - trunc(a/b) * b

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;
    sfpi::vFloat b_abs = sfpi::abs(b);

    // Compute reciprocal 1/b
    sfpi::vFloat recip = ckernel::sfpu::_sfpu_reciprocal_<2>(b);

    // Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * recip;

    // Compute trunc(a/b)
    sfpi::l_reg[sfpi::LRegs::LReg0] = div_result;
    _trunc_body_();
    sfpi::vFloat trunc_div = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat tmp2 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat tmp3 = sfpi::l_reg[sfpi::LRegs::LReg3];

    // Compute fmod = a - trunc(a/b) * b
    sfpi::vFloat result = a - trunc_div * b;

    // Post-correction - fmod result must satisfy |result| < |b|
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

    // Sign correction - fmod result must have same sign as 'a' (or be zero)
    // If a > 0 and result < 0, the truncation was 1 too high, need to add b
    // If a < 0 and result > 0, the truncation was 1 too low, need to subtract b
    // This fixes cases where a/b ≈ 0.9999999 but rounds to 1 due to reciprocal error
    v_if(a >= sfpi::vFloat(0.0f)) {
        // a is positive, result should be >= 0
        v_if(result < sfpi::vFloat(0.0f)) {
            result = result + b_abs;  // over-truncated
        }
        v_endif;
    }
    v_else {
        // a is negative, result should be <= 0
        v_if(result > sfpi::vFloat(0.0f)) {
            result = result - b_abs;  // under-truncated
        }
        v_endif;
    }
    v_endif;

    // Handle special cases using conditional assignment (NOT early return!)
    // When a == b, fmod(a, b) = 0
    v_if(a == b) { result = sfpi::vFloat(0.0f); }
    v_endif;

    // Handle division by zero - return NaN
    v_if(b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, 0));
    }

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_fmod_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_fmod_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_fmod(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_fmod_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void fmod_int32_init() {
    div_floor_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
inline void fmod_binary_init() {
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
