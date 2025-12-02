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

    // SFPI tries to use MOD0_FMT_INT32_SM, which interprets values as
    // sign-magnitude integers, and is deprecated on Blackhole.  Instead, we
    // want to use MOD0_FMT_INT32=4, which gives us the original two's
    // complement integers.

    // Equivalent to: sfpi::vUInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
    sfpi::vUInt b = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].get());

    // When converting to float, integers are treated as sign-magnitude.
    // Convert inputs to positive values to avoid conversion problems; the
    // original inputs are two's complement integers.  Note that
    // sfpi::abs(-2**31) will return -2**31, which will give -0.0 when
    // converted to float via sfpi::int32_to_float.
    b = sfpi::abs(b);

    // Convert to floats, but check for the edge case mentioned above.
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(b_f < 0.0f) { b_f = 2147483648.0f; }
    v_endif;

    // Compute 1/b accurate to ~21 bits of precision via:
    // 1. Linear approximation: ~3 bits
    // 2. Newton-Raphson: ~7 bits
    // 3. Halley's Method: ~21 bits

    // We interleave SFPMAD with the loading and conversion of `a`.

    // Combines the sign and exponent of -1.0 with the mantissa of `b_f`.
    // Scale the input value to the range [1.0, 2.0), and make it negative.
    sfpi::vFloat neg_b_f = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(b_f));
    // Linear approximation.
    sfpi::vFloat inv_b_f = sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1 * neg_b_f;
    sfpi::vFloat scale = sfpi::setman(b_f, 0);

    // Newton-Raphson
    sfpi::vFloat t = inv_b_f * neg_b_f + sfpi::vConst1;
    scale = sfpi::reinterpret<sfpi::vFloat>((254 << 23) - sfpi::reinterpret<sfpi::vInt>(scale));
    inv_b_f = t * inv_b_f + inv_b_f;

    // Halley's Method
    sfpi::vFloat e = inv_b_f * neg_b_f + sfpi::vConst1;

    // Equivalent to: sfpi::vUInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vUInt a = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());

    // Continue Halley's Method
    e = e * e + e;
    a = sfpi::abs(a);

    // Final step of Halley's Method
    inv_b_f = e * inv_b_f + inv_b_f;
    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);

    // Apply scale
    inv_b_f = inv_b_f * scale;
    v_if(a_f < 0.0f) { a_f = 2147483648.0f; }
    v_endif;

    // Initial approximation of quotient: q = a * 1/b.
    // We add a special mantissa alignment factor 2.0f**(23+11), which shifts
    // the mantissa so that we extract the top 21 bits of the result.
    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;

    sfpi::vUInt MASK_11 = 0x7ff;
    sfpi::vUInt q = sfpi::exman9(q_f);

    // Compute qb = q * b.  This tells us how close our approximation `q` is to
    // the target `a`.  Note: we only care about the top ~21 bits.
    // Keep the top 21 bits of 32-bit q: q_hi = q>>11
    // Now q2 = q>>22, q1 = q>>11
    // And so qb = (q2<<22 + q1<<11) * (b2<<22 + b1<<11 + b0)
    //           = (q2<<22 * b0) + (q1<<11 * b1<<11) + (q1<<11 * b0)
    sfpi::vFloat q1 = int32_to_float(q & MASK_11, 0);
    sfpi::vFloat q2 = int32_to_float(q >> 11, 0);
    sfpi::vFloat b1 = int32_to_float((b >> 11) & MASK_11, 0);
    sfpi::vFloat b0 = int32_to_float(b & MASK_11, 0);
    q = q << 11;

    sfpi::vFloat MANTISSA_ALIGNMENT_OFFSET = 8388608.0f;
    sfpi::vFloat hi = q2 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat lo = q1 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    hi = q1 * b1 + hi;

    sfpi::vInt qb = sfpi::exman9(lo) << 11;
    qb += sfpi::exman9(hi) << 22;

    // Compute remainder.
    // a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    a = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());
    a = sfpi::abs(a);
    sfpi::vInt r = a - qb;
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);

    // Compute correction value in float32.
    sfpi::vFloat correction_f = r_f * inv_b_f;
    sfpi::vFloat b2 = sfpi::int32_to_float(b >> 22, 0);
    sfpi::vInt correction = sfpi::float_to_uint16(correction_f, 0);
    correction_f = sfpi::int32_to_float(correction, 0);

    // correction should fit into 11 bits, thus:
    // tmp = correction * (b2<<22 + b1<<11 + b0)

    sfpi::vFloat low = correction_f * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat mid = correction_f * b1 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat top = correction_f * b2 + MANTISSA_ALIGNMENT_OFFSET;

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
    // Reload signed values here due to register pressure.
    // a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    // b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
    a = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());
    b = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].get());
    sfpi::vInt sign = a ^ b;
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

    // sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
    __builtin_rvtt_sfpstore(
        result.get(), 4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].get());
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
    sfpi::vConstFloatPrgm0 = 17179869184.0f;
    sfpi::vConstFloatPrgm1 = 0.470588266849517822265625f;
    sfpi::vConstFloatPrgm2 = 1.41176474094390869140625f;
}

template <bool APPROXIMATION_MODE>
inline void div_floor_init() {
    div_trunc_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
