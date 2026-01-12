// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_div_int32_floor.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Remainder = a - floor(a / b) * b
// Use 32-bit integer division from ckernel_sfpu_div_int32_floor.h
sfpi_inline void calculate_remainder_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Equivalent to: sfpi::vUInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
    sfpi::vUInt b = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].get());

    // When converting to float, integers are treated as sign-magnitude.
    // Convert inputs to positive values to avoid conversion problems.
    b = sfpi::abs(b);

    // Convert to float for reciprocal computation
    // Handle edge case: if conversion results in negative
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(b_f < 0.0f) { b_f = 2147483648.0f; }
    v_endif;

    // Compute reciprocal of b
    sfpi::vFloat neg_b_f = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(b_f));

    sfpi::vFloat inv_b_f = sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1 * neg_b_f;

    sfpi::vFloat scale = sfpi::setman(b_f, 0);

    // First Newton-Raphson iteration: inv_b_f = inv_b_f * (2 - inv_b_f * b_f)
    sfpi::vFloat t = inv_b_f * neg_b_f + sfpi::vConst1;
    scale = sfpi::reinterpret<sfpi::vFloat>((254 << 23) - sfpi::reinterpret<sfpi::vInt>(scale));
    inv_b_f = t * inv_b_f + inv_b_f;

    // Halley's Method
    sfpi::vFloat e = inv_b_f * neg_b_f + sfpi::vConst1;

    // Equivalent to: sfpi::vUInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vUInt a = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());

    a = sfpi::abs(a);

    // Continue Halley's Method
    e = e * e + e;
    inv_b_f = e * inv_b_f + inv_b_f;

    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);
    v_if(a_f < 0.0f) { a_f = 2147483648.0f; }
    v_endif;

    // Apply scaling factor to finalize reciprocal
    inv_b_f = inv_b_f * scale;

    // Initial quotient approximation : q = a * 1/b
    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;
    sfpi::vUInt q = sfpi::exman9(q_f);

    sfpi::vUInt MASK_11 = 0x7ff;

    // Split q and b into 11-bit chunks to compute q * b
    sfpi::vFloat q1 = int32_to_float(q & MASK_11, 0);          // Lower 11 bits of q
    sfpi::vFloat q2 = int32_to_float(q >> 11, 0);              // Upper bits of q
    sfpi::vFloat b1 = int32_to_float((b >> 11) & MASK_11, 0);  // Middle 11 bits of b
    sfpi::vFloat b0 = int32_to_float(b & MASK_11, 0);          // Lower 11 bits of b
    q = q << 11;

    sfpi::vFloat MANTISSA_ALIGNMENT_OFFSET = 8388608.0f;

    // hi = q2 * b0 + q1 * b1 (high part)
    // lo = q1 * b0 (low part)
    sfpi::vFloat hi = q2 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat lo = q1 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    hi = q1 * b1 + hi;

    sfpi::vInt qb = sfpi::exman9(lo) << 11;
    qb += sfpi::exman9(hi) << 22;

    // Compute remainder
    a = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());
    a = sfpi::abs(sfpi::reinterpret<sfpi::vInt>(a));
    sfpi::vInt r = a - qb;
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);

    // Compute correction: r / b in float32
    sfpi::vFloat correction_f = r_f * inv_b_f;
    sfpi::vFloat b2 = sfpi::int32_to_float(b >> 22, 0);
    sfpi::vInt correction = sfpi::float_to_uint16(correction_f, 0);
    correction_f = sfpi::int32_to_float(correction, 0);

    // tmp = correction * (b2<<22 + b1<<11 + b0)
    sfpi::vFloat low = correction_f * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat mid = correction_f * b1 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat top = correction_f * b2 + MANTISSA_ALIGNMENT_OFFSET;

    sfpi::vInt tmp = sfpi::exman9(low);
    tmp += sfpi::exman9(mid) << 11;
    tmp += sfpi::exman9(top) << 22;

    // Adjust remainder based on its sign
    v_if(r < 0) { r += tmp; }
    v_else { r -= tmp; }
    v_endif;

    // Since the correction might have been rounded, we may need to correct one
    // additional bit.  The (r - 1) < 0 check is required to handle r=INT_MIN.
    v_if(r < 0 && (r - 1) < 0) { r += b; }
    v_elseif(r >= b) { r -= b; }
    v_endif;

    // Reload original signed values for sign handling
    // Equivalent to: sfpi::vUInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt a_signed = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].get());
    // Equivalent to: sfpi::vUInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
    sfpi::vInt b_signed = __builtin_rvtt_sfpload(
        4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].get());

    sfpi::vInt sign = a_signed ^ b_signed;
    v_if(r != 0) {
        v_if(sign < 0) {  // signs differ: need to adjust for floor division
            // When signs differ, floor(a/b) = trunc(a/b) - 1, so remainder needs adjustment
            v_if(a_signed < 0) { r = b_signed - r; }
            v_else { r += b_signed; }
            v_endif;
        }
        v_elseif(a_signed < 0 && b_signed < 0) { r = -r; }
        v_endif;
    }
    v_endif;

    // Equivalent to: sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
    __builtin_rvtt_sfpstore(
        r.get(), 4, sfpi::SFPLOAD_ADDR_MODE_NOINC, sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].get());
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
