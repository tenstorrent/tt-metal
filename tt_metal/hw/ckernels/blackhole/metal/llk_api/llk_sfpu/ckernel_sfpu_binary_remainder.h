// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_div_int32_floor.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// 2^31 as float (used for INT32 sign-magnitude conversion edge cases)
constexpr float TWO_POW_31 = 2147483648.0f;

// Computes the unsigned remainder: |a| - floor(|a| / |b|) * |b|
// Use 32-bit integer division from ckernel_sfpu_div_int32_floor.h
// Returns: unsigned remainder r
sfpi_inline sfpi::vInt compute_unsigned_remainder_int32(const sfpi::vInt& a_signed, const sfpi::vInt& b_signed) {
    // Get absolute values for unsigned remainder computation
    sfpi::vInt a = sfpi::abs(a_signed);
    sfpi::vInt b = sfpi::abs(b_signed);

    // Convert to float for reciprocal computation
    // Handle 2^31 edge case where sign-magnitude conversion yields negative
    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
    v_endif;
    v_if(b_f < 0.0f) { b_f = TWO_POW_31; }
    v_endif;

    // Compute reciprocal of b using a single Newton–Raphson refinement
    // Accuracy is sufficient because we apply an integer correction later.
    sfpi::vFloat inv_b_f = sfpi::approx_recip(b_f);
    // One NR step: inv_b = inv_b * (2 - b * inv_b)
    sfpi::vFloat e = -inv_b_f * b_f + sfpi::vConst1;
    inv_b_f = e * inv_b_f + inv_b_f;

    // Initial quotient approximation: q = a * (1/b)
    sfpi::vFloat q_f = a_f * inv_b_f + vConstFloatPrgm0;
    sfpi::vUInt q = sfpi::exman9(q_f);

    // Compute q * b using 24-bit multiplication
    sfpi::vInt qb = __builtin_rvtt_sfpmul24(q.get(), b.get(), 0);
    qb <<= 10;

    // Compute initial remainder
    sfpi::vInt r = a - qb;

    // Compute correction for approximation error: correction = |r| / b
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);
    sfpi::vInt correction = sfpi::float_to_uint16(r_f * inv_b_f, 0);

    // Compute correction * b (full 32-bit result from 24-bit multiplies)
    sfpi::vInt tmp_lo = __builtin_rvtt_sfpmul24(correction.get(), b.get(), 0);
    sfpi::vInt tmp_hi = __builtin_rvtt_sfpmul24(correction.get(), b.get(), 1);
    sfpi::vInt b_hi = b >> 23;
    b_hi = __builtin_rvtt_sfpmul24(correction.get(), b_hi.get(), 0);
    sfpi::vInt tmp = tmp_lo + ((tmp_hi + b_hi) << 23);

    // Extract sign mask of r
    // r_sign = 0 if r >= 0, -1 if r < 0
    sfpi::vInt r_sign = r >> 31;

    // Apply correction with sign of r
    // If r < 0  -> r += tmp
    // Else      -> r -= tmp
    sfpi::vInt signed_tmp = (tmp ^ r_sign) - r_sign;
    r -= signed_tmp;

    // Final adjustment to ensure r is in [0, b)
    v_if(r < 0 && (r - 1) < 0) { r += b; }
    v_elseif(r >= b) { r -= b; }
    v_endif;

    return r;
}

// Remainder = a - floor(a / b) * b
sfpi_inline void calculate_remainder_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Read inputs
    sfpi::vInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // Remainder sign handling
    v_if(a_signed < 0) { r = -r; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = r;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_remainder_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // remainder(a, b) = a - floor(a/b) * b

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;

    // Compute reciprocal 1/b
    sfpi::vFloat recip = ckernel::sfpu::_sfpu_reciprocal_<2>(b);

    // Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * recip;

    // Compute floor(a/b)
    // Input in LReg0, output in LReg1. LReg2/LReg3 are clobbered by _floor_body_(),
    // so we must read them to inform the SFPI register allocator they are not immediately available.
    sfpi::l_reg[sfpi::LRegs::LReg0] = div_result;
    _floor_body_();
    sfpi::vFloat floor_div = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vFloat _lreg2_clobbered_ = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vFloat _lreg3_clobbered_ = sfpi::l_reg[sfpi::LRegs::LReg3];

    // Compute remainder = a - floor(a/b) * b
    sfpi::vFloat result = a - floor_div * b;

    // Sign correction: remainder must match the sign of b (or be zero).
    // XOR of the float bit-patterns detects sign mismatch via the MSB,
    // avoiding a compound conditional with four comparisons and an OR.
    v_if(
        result != 0.0f && (((sfpi::reinterpret<sfpi::vUInt>(result) ^ sfpi::reinterpret<sfpi::vUInt>(b)) >> 31) != 0)) {
        result += b;
    }
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
inline void calculate_remainder_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = false>
inline void calculate_sfpu_binary_remainder(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat result = _sfpu_binary_remainder_<is_fp32_dest_acc_en>(in0, in1);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void remainder_int32_init() {
    div_floor_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void remainder_binary_init() {
    _init_reciprocal_<APPROXIMATION_MODE, is_fp32_dest_acc_en, false>();
}

}  // namespace ckernel::sfpu
