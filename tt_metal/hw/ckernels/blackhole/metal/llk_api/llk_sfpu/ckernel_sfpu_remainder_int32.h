// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_div_int32_floor.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Computes the unsigned remainder: |a| - floor(|a| / |b|) * |b|
// Use 32-bit integer division from ckernel_sfpu_div_int32_floor.h
// Returns: r (unsigned remainder), a_signed, b_signed (original signed values)
sfpi_inline void compute_unsigned_remainder_int32(
    const uint dst_index_in0, const uint dst_index_in1, sfpi::vInt& r, sfpi::vInt& a_signed, sfpi::vInt& b_signed) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // When converting to float, integers are treated as sign-magnitude.
    // Convert inputs to positive values to avoid conversion problems.
    sfpi::vInt b = sfpi::abs(b_signed);

    // Convert to float for reciprocal computation
    // Handle edge case: if conversion results in negative
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(b_f < 0.0f) { b_f = 2147483648.0f; }
    v_endif;

    // Compute reciprocal of b
    sfpi::vFloat inv_b_f = sfpi::approx_recip(b_f);
    sfpi::vFloat e = -inv_b_f * b_f + sfpi::vConst1;
    a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    e = e * e + e;
    sfpi::vInt a = sfpi::abs(a_signed);
    inv_b_f = e * inv_b_f + inv_b_f;
    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);
    v_if(a_f < 0.0f) { a_f = 2147483648.0f; }
    v_endif;

    // Initial quotient approximation : q = a * 1/b
    sfpi::vFloat q_f = a_f * inv_b_f + vConstFloatPrgm0;
    sfpi::vUInt q = sfpi::exman9(q_f);

    // Compute q * b
    // SFPMUL24 multiplies two 23-bit integers
    sfpi::vInt qb;

    qb.get() = __builtin_rvtt_bh_sfpmul24(q.get(), b.get(), 0);

    q <<= 10;
    qb <<= 10;

    // Compute remainder
    r = a - qb;
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);

    // Compute correction: r / b in float32
    sfpi::vFloat correction_f = r_f * inv_b_f;
    sfpi::vInt b1 = b >> 23;
    sfpi::vInt correction = sfpi::float_to_uint16(correction_f, 0);

    // Compute tmp = correction * b
    sfpi::vInt tmp_hi;
    sfpi::vInt tmp_lo;
    b1.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b1.get(), 0);
    tmp_hi.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b.get(), 1);
    tmp_lo.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b.get(), 0);
    tmp_hi += b1;
    tmp_hi <<= 23;
    sfpi::vInt tmp = tmp_lo + tmp_hi;

    // Adjust remainder based on its sign
    v_if(r < 0) { r += tmp; }
    v_else { r -= tmp; }
    v_endif;

    // Since the correction might have been rounded, we may need to correct one
    // additional bit.  The (r - 1) < 0 check is required to handle r=INT_MIN.
    v_if(r < 0 && (r - 1) < 0) { r += b; }
    v_elseif(r >= b) { r -= b; }
    v_endif;
}

// Remainder = a - floor(a / b) * b
// Use 32-bit integer division from ckernel_sfpu_div_int32_floor.h
sfpi_inline void calculate_remainder_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Compute unsigned remainder
    sfpi::vInt r;
    sfpi::vInt a_signed;
    sfpi::vInt b_signed;
    compute_unsigned_remainder_int32(dst_index_in0, dst_index_in1, r, a_signed, b_signed);

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
#pragma GCC unroll 8
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
