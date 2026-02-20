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
    // Get absolute value of b for reciprocal computation
    sfpi::vUInt b = sfpi::abs(b_signed);

    // Convert to float for reciprocal computation
    // Handle edge case: if conversion results in negative
    sfpi::vFloat b_f = sfpi::int32_to_float(b, 0);
    v_if(b_f < 0.0f) { b_f = TWO_POW_31; }
    v_endif;

    // Compute reciprocal of b
    sfpi::vFloat neg_b_f = sfpi::setman(sfpi::vConstNeg1, sfpi::reinterpret<sfpi::vInt>(b_f));

    sfpi::vFloat inv_b_f = sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1 * neg_b_f;

    sfpi::vFloat scale = sfpi::setman(b_f, 0);

    // First Newton-Raphson iteration: inv_b_f = inv_b_f * (2 - inv_b_f * b_f)
    sfpi::vFloat t = inv_b_f * neg_b_f + sfpi::vConst1;
    scale = sfpi::reinterpret<sfpi::vFloat>((254 << 23) - sfpi::reinterpret<sfpi::vInt>(scale));
    inv_b_f = t * inv_b_f + inv_b_f;

    // Second Newton-Raphson iteration (interleaved with abs(a) computation)
    sfpi::vFloat e = inv_b_f * neg_b_f + sfpi::vConst1;
    sfpi::vUInt a = sfpi::abs(a_signed);
    inv_b_f = e * inv_b_f + inv_b_f;

    sfpi::vFloat a_f = sfpi::int32_to_float(a, 0);
    v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
    v_endif;

    // Apply scaling factor to finalize reciprocal
    inv_b_f = inv_b_f * scale;

    // Initial quotient approximation : q = a * 1/b
    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;
    sfpi::vUInt q = sfpi::exman9(q_f);

    // Recompute b for chunk extraction to reduce register pressure
    b = sfpi::abs(b_signed);

    // 8388608.0f = 2^23 is used as a Bias for mantissa alignment
    sfpi::vFloat MANTISSA_ALIGNMENT_OFFSET = 8388608.0f;

    // Split q and b into 11-bit chunks to compute q * b
    sfpi::vUInt MASK_11 = 0x7ff;
    sfpi::vFloat q1 = int32_to_float(q & MASK_11, 0);
    sfpi::vFloat q2 = int32_to_float(q >> 11, 0);
    sfpi::vFloat b1 = int32_to_float((b >> 11) & MASK_11, 0);
    sfpi::vFloat b0 = int32_to_float(b & MASK_11, 0);

    // hi = q2 * b0 + q1 * b1 (high part)
    // lo = q1 * b0 (low part)
    sfpi::vFloat hi = q2 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat lo = q1 * b0 + MANTISSA_ALIGNMENT_OFFSET;
    hi = q1 * b1 + hi;

    sfpi::vInt qb = sfpi::exman9(lo) << 11;
    qb += sfpi::exman9(hi) << 22;

    // Compute remainder - recompute abs(a_signed)
    a = sfpi::abs(a_signed);
    sfpi::vInt r = a - qb;

    // Use abs(r) for correction computation
    sfpi::vFloat r_f = sfpi::int32_to_float(sfpi::abs(r), 0);

    // Compute correction: r / b in float32
    sfpi::vFloat correction_f = r_f * inv_b_f;
    sfpi::vInt correction = sfpi::float_to_uint16(correction_f, 0);
    correction_f = sfpi::int32_to_float(correction, 0);

    // Recompute b chunks for correction multiplication to reduce register pressure
    b = sfpi::abs(b_signed);
    b0 = int32_to_float(b & MASK_11, 0);
    b1 = int32_to_float((b >> 11) & MASK_11, 0);
    sfpi::vFloat b2 = sfpi::int32_to_float(b >> 22, 0);

    // tmp = correction * (b2<<22 + b1<<11 + b0)
    sfpi::vFloat low = correction_f * b0 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat mid = correction_f * b1 + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat top = correction_f * b2 + MANTISSA_ALIGNMENT_OFFSET;

    sfpi::vInt tmp = sfpi::exman9(low);
    tmp += sfpi::exman9(mid) << 11;
    tmp += sfpi::exman9(top) << 22;

    // Extract sign mask of r
    // r_sign = 0 if r >= 0, -1 if r < 0
    sfpi::vInt r_sign = sfpi::reinterpret<sfpi::vInt>(sfpi::reinterpret<sfpi::vUInt>(r) >> 31);
    r_sign = -r_sign;

    // Apply correction with sign of r
    // If r < 0  -> r += tmp
    // Else      -> r -= tmp
    sfpi::vInt signed_tmp = (tmp ^ r_sign) - r_sign;
    r -= signed_tmp;

    // Final adjustment - recompute b to reduce register pressure
    b = sfpi::abs(b_signed);
    v_if(r < 0 && (r - 1) < 0) { r += b; }
    v_elseif(r >= b) { r -= b; }
    v_endif;

    return r;
}

// Remainder = a - floor(a / b) * b
sfpi_inline void calculate_remainder_int32_body(
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
