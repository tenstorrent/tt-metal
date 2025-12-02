// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// 32-bit integer division.
// Template parameter `floor` indicates whether "floor" (true) or "trunc"
// (false) rounding mode should be used.
template <bool floor>
sfpi_inline void calculate_div_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    sfpi::vInt b_orig = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // When converting to float, the integers are treated as sign-magnitude.
    // Convert inputs to positive values to avoid conversion problems, as the
    // original inputs are two's complement integers.  Note that
    // sfpi::abs(-2**31) will return -2**31, which will give -0.0 when
    // converted to float via sfpi::int32_to_float.
    sfpi::vInt b = sfpi::abs(b_orig);

    // Convert to floats, but check for the edge case mentioned above.
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(b_f < 0.0f) { b_f = 2147483648.0f; }
    v_endif;

    // Compute 1/b accurate to ~22 bits of precision via Halley's Method.
    // Since the inputs can be as large as 2**31-1, this only gives us an
    // initial approximation.
    // We interleave SFPMAD with the loading and conversion of `a`.
    sfpi::vFloat inv_b_f = sfpi::approx_recip(b_f);
    sfpi::vFloat e = -inv_b_f * b_f + sfpi::vConst1;
    sfpi::vInt a_orig = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    e = e * e + e;
    sfpi::vInt a = sfpi::abs(a_orig);
    inv_b_f = e * inv_b_f + inv_b_f;
    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);
    v_if(a_f < 0.0f) { a_f = 2147483648.0f; }
    v_endif;

    // Initial approximation q = a * 1/b.
    // We add a special mantissa alignment factor 2.0f**(23+10), which shifts
    // the mantissa so that we extract the top 22 bits of the result.
    sfpi::vFloat q_f = a_f * inv_b_f + vConstFloatPrgm0;
    sfpi::vInt sign = a_orig ^ b_orig;
    sfpi::vUInt q = sfpi::exman9(q_f);

    // Compute qb = q * b.  This tells us how close our approximation `q` is to
    // the target `a`.  We split into 23-bit chunks.
    // Since inv_b is only accurate to ~22 bits, we only care about the upper
    // 22 bits, so we can compute qb = (q1<<10 + 0) * (b1<<22 + b0)
    //                               = (q1<<10) * b0

    sfpi::vInt qb;

    // Despite the name, SFPMUL24 multiplies two 23-bit integers, giving the
    // low or high 23 bits of the product (last argument: 0=lo, 1=hi).  Inputs
    // do not need to be masked as this is done internally.
    qb.get() = __builtin_rvtt_bh_sfpmul24(q.get(), b.get(), 0);

    q <<= 10;
    qb <<= 10;

    // Compute remainder.
    sfpi::vInt r = a - qb;
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);

    // Compute correction value in float32.
    sfpi::vFloat correction_f = r_f * inv_b_f;
    sfpi::vInt b1 = b >> 23;
    sfpi::vInt correction = sfpi::float_to_uint16(correction_f, 0);

    // Compute tmp = correction * b.
    sfpi::vInt tmp_hi;
    sfpi::vInt tmp_lo;
    b1.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b1.get(), 0);
    tmp_hi.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b.get(), 1);
    tmp_lo.get() = __builtin_rvtt_bh_sfpmul24(correction.get(), b.get(), 0);
    tmp_hi += b1;
    tmp_hi <<= 23;
    sfpi::vInt tmp = tmp_lo + tmp_hi;

    // Apply correction and adjust remainder.
    v_if(r < 0) {
        q -= correction;
        r += tmp;
    }
    v_else {
        q += correction;
        r -= tmp;
    }
    v_endif;

    // Since the correction might have been rounded, we may need to correct one
    // additional bit.  The (r - 1) < 0 check is required to handle r=INT_MIN.
    v_if(r < 0 && (r - 1) < 0) {
        q -= 1;
        r += b;
    }
    v_elseif(r >= b) {
        q += 1;
        r -= b;
    }
    v_endif;

    sfpi::vInt result = q;

    // If a ^ b >= 0, then the result will be positive, otherwise negative.
    // Finally, if we expect a negative result, negate the value (two's complement).
    v_if(sign < 0) {
        result = -result;

        // Optionally, if we want "floor" rounding, check for a remainder
        // and subtract one for negative numbers, to round towards negative
        // infinity.

        if constexpr (floor) {
            v_if(r != 0) { result -= 1; }
            v_endif;
        }
    }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32_floor(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_div_int32_body<true>(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32_trunc(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_div_int32_body<false>(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void div_trunc_init() {
    sfpi::vConstFloatPrgm0 = 8589934592.0f;
}

template <bool APPROXIMATION_MODE>
inline void div_floor_init() {
    div_trunc_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
