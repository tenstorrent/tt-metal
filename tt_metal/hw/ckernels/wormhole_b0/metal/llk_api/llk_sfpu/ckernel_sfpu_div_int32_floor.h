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

    // When converting to float, integers are treated as sign-magnitude.
    // Convert inputs to positive values to avoid conversion problems; the
    // original inputs are two's complement integers.  Note that
    // sfpi::abs(-2**31) will return -2**31, which will give -0.0 when
    // converted to float via sfpi::int32_to_float.
    a = sfpi::abs(a);
    b = sfpi::reinterpret<sfpi::vInt>(sfpi::abs(b));

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

    // Compute qb = q * b.  This tells us how close our approximation `q` is to
    // the target `a`.  We split into 23-bit chunks.
    // Idea: maybe just keep the top 22 bits of 31-bit q: q_hi = q>>9
    // Now q2 = q>>20, q1 = q>>9
    // And so qb = (q2<<20 + q1<<9) * (b2<<22 + b1<<11 + b0)
    //           = (q2<<20 * b0) + (q1<<9 * b1<<11) + (q1<<9 * b0)

    q = q >> 9;
    sfpi::vFloat q1 = int32_to_float(q & sfpi::vConstIntPrgm2, 0);
    sfpi::vFloat q2 = int32_to_float(q >> 11, 0);
    sfpi::vFloat b1 = int32_to_float((b >> 11) & sfpi::vConstIntPrgm2, 0);
    sfpi::vFloat b0 = int32_to_float(b & sfpi::vConstIntPrgm2, 0);
    q = q << 9;

    sfpi::vFloat lo = q1 * b0 + sfpi::vConstFloatPrgm1;
    sfpi::vFloat hi = q2 * b0 + sfpi::vConstFloatPrgm1;
    hi = q1 * b1 + hi;

    sfpi::vInt qb = sfpi::exman9(hi) << 20;
    qb += sfpi::exman9(lo) << 9;

    // Compute remainder.
    a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    a = sfpi::abs(a);
    sfpi::vInt r = a - qb;
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);

    // Compute correction value in float32.
    sfpi::vFloat correction_f = r_f * inv_b_f;
    sfpi::vInt correction = sfpi::float_to_uint16(correction_f, 0);
    correction_f = sfpi::int32_to_float(correction, 0);

    // correction should fit into 11 bits, thus:
    // tmp = correction * (b2<<22 + b1<<11 + b0)

    sfpi::vFloat b2 = sfpi::int32_to_float(b >> 22);
    sfpi::vFloat low = correction_f * b0 + sfpi::vConstFloatPrgm1;
    sfpi::vFloat mid = correction_f * b1 + sfpi::vConstFloatPrgm1;
    sfpi::vFloat top = correction_f * b2 + sfpi::vConstFloatPrgm1;

    sfpi::vInt tmp = sfpi::exman9(low);
    tmp += sfpi::exman9(mid) << 11;
    tmp += sfpi::exman9(top) << 22;

    v_if(r < 0) {
        q -= correction;
        r += tmp;
    }
    v_else {
        q += correction;
        r -= tmp;
    }
    v_endif;

    v_if(r < 0) {
        q -= 1;
        r += b;
    }
    v_elseif(r >= b) {
        q += 1;
        r -= b;
    }
    v_endif;

    // If a ^ b >= 0, then the result will be positive, otherwise negative.
    a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
    sfpi::vInt sign = a ^ b;
    // Finally, if we expect a negative result, negate the value (two's complement).
    v_if(sign < 0) {
        q = -q;

        // Optionally, if we want "floor" rounding, check for a remainder
        // and subtract one for negative numbers, to round towards negative
        // infinity.

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
    sfpi::vConstFloatPrgm1 = 8388608.0f;
}

template <bool APPROXIMATION_MODE>
inline void div_trunc_init() {
    _init_sfpu_reciprocal_<false>();
    sfpi::vConstFloatPrgm1 = 8388608.0f;
    sfpi::vConstIntPrgm2 = 0x7ff;
}

}  // namespace ckernel::sfpu
