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

    sfpi::vInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // If a ^ b >= 0, then the result will be positive, otherwise negative.
    sfpi::vInt sign = a ^ b;

    // When converting to float, the integers are treated as sign-magnitude.
    // Convert inputs to positive values to avoid conversion problems, as the
    // original inputs are two's complement integers.  Note that
    // sfpi::abs(-2**31) will return -2**31, which will give -0.0 when
    // converted to float via sfpi::int32_to_float.
    a = sfpi::abs(a);
    b = sfpi::abs(b);

    // Convert to floats, but check for the edge case mentioned above.
    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);
    v_if(a_f < 0.0f) { a_f = 2147483648.0f; }
    v_endif;
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(b_f < 0.0f) { b_f = 2147483648.0f; }
    v_endif;

    // This is accurate to ~23 bits of precision.  Since the inputs can be as
    // large as 2**31-1, so this only gives us an initial approximation.
    sfpi::vFloat inv_b_f = _sfpu_reciprocal_<2>(b_f);

    // Initial approximation q = a * 1/b.
    sfpi::vFloat q_f = a_f * inv_b_f;

    // Convert from float to int32, truncating any fractional parts.  No sign
    // check is necessary as q will always be positive, due to using abs(a) and
    // abs(b).
    sfpi::vInt q = 0;
    sfpi::vInt exp = sfpi::exexp(q_f);
    v_if(exp >= 0) {
        q = sfpi::exman8(q_f);
        exp = exp - 23;
        q = q << exp;
    }
    v_endif;
    sfpi::vInt q0 = q;

    // Compute qb = q * b.  This tells us how close our approximation `q` is to
    // the target `a`.  We split into 23-bit chunks.

    sfpi::vInt q1 = q >> 23;
    sfpi::vInt b1 = b >> 23;
    sfpi::vInt lo;
    sfpi::vInt hi;

    // Despite the name, SFPMUL24 multiplies two 23-bit integers, giving the
    // low or high 23 bits of the product (last argument: 0=lo, 1=hi).  Inputs
    // do not need to be masked as this is done internally.
    q1.get() = __builtin_rvtt_bh_sfpmul24(b.get(), q1.get(), 0);
    b1.get() = __builtin_rvtt_bh_sfpmul24(q.get(), b1.get(), 0);
    lo.get() = __builtin_rvtt_bh_sfpmul24(q.get(), b.get(), 0);
    hi.get() = __builtin_rvtt_bh_sfpmul24(q.get(), b.get(), 1);

    q1 += b1;
    q1 += hi;

    // This is qb.
    lo += q1 << 23;

    // Compute remainder.  Note that since our initial approximation has ~23
    // bits of precision, we don't expect the remainder to be larger than ~8
    // bits.
    sfpi::vInt r = a - lo;

    // Conversion to float is lossless as r should be ~8 bits.
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);

    // Compute correction value in float32.
    sfpi::vFloat correction_f = r_f * inv_b_f;

    // Convert to integer, truncating the fractional part.
    sfpi::vInt correction = 0;
    exp = sfpi::exexp(correction_f);
    v_if(exp >= 0) {
        correction = sfpi::exman8(correction_f);
        exp = exp - 23;
        correction = correction << exp;
    }
    v_endif;

    // The correction value could be negative.
    v_if(r < 0) { correction = ~correction; }
    v_endif;

    // Apply correction.
    q += correction;

    // Finally, if we expect a negative result, negate the value (two's complement).
    v_if(sign < 0) {
        q = -q;

        // Optionally, if we want "floor" rounding, check if rounding was
        // applied and subtract one for negative numbers, to round towards
        // negative infinity.

        if constexpr (floor) {
            v_if(r != 0) { q -= 1; }
            v_endif;
        }
    }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = q;
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
inline void div_floor_init() {
    _init_sfpu_reciprocal_<false>();
}

template <bool APPROXIMATION_MODE>
inline void div_trunc_init() {
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
