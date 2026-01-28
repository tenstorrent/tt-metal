// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    sfpi::vInt qb;
    qb.get() = __builtin_rvtt_bh_sfpmul24(q.get(), b.get(), 0);
    qb <<= 10;

    // Compute initial remainder
    sfpi::vInt r = a - qb;

    // Compute correction for approximation error: correction = |r| / b
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);
    sfpi::vInt correction = sfpi::float_to_uint16(r_f * inv_b_f, 0);

    // Compute correction * b (full 32-bit result from 24-bit multiplies)
    sfpi::vInt tmp_lo, tmp_hi, b_hi;
    tmp_lo.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b.get(), 0);
    tmp_hi.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b.get(), 1);
    b_hi = b >> 23;
    b_hi.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b_hi.get(), 0);
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
    sfpi::vInt sign = a_signed ^ b_signed;
    v_if(r != 0) {
        v_if(sign < 0) {
            // When signs differ, floor(a/b) = trunc(a/b) - 1, so remainder needs adjustment
            v_if(a_signed < 0) { r = b_signed - r; }
            v_else { r += b_signed; }
            v_endif;
        }
        v_elseif(a_signed < 0 && b_signed < 0) { r = -r; }
        v_endif;
    }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = r;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_remainder_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void remainder_int32_init() {
    div_floor_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
