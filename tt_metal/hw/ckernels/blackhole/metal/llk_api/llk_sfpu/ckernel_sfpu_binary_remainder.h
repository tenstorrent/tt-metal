// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_div_int32_floor.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"

namespace ckernel::sfpu {

// 2^31 as float (used for INT32 sign-magnitude conversion edge cases)
constexpr float TWO_POW_31 = 2147483648.0f;

// Computes the unsigned remainder: |a| - floor(|a| / |b|) * |b|
// Use 32-bit integer division from ckernel_sfpu_div_int32_floor.h
// Returns: unsigned remainder r
sfpi_inline sfpi::vInt compute_unsigned_remainder_int32(const sfpi::vInt& a_signed, const sfpi::vInt& b_signed) {
    // Get absolute values for unsigned remainder computation
    sfpi::vMag a = sfpi::abs(a_signed);
    sfpi::vMag b = sfpi::abs(b_signed);

    // Convert to float for reciprocal computation
    // Handle 2^31 edge case where sign-magnitude conversion yields negative
    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest);
    sfpi::vFloat b_f = sfpi::convert<sfpi::vFloat>(b, sfpi::RoundMode::Nearest);
    v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
    v_endif;
    v_if(b_f < 0.0f) { b_f = TWO_POW_31; }
    v_endif;

    // Compute reciprocal of b using a single Newton–Raphson refinement
    // Accuracy is sufficient because we apply an integer correction later.
    sfpi::vFloat inv_b_f = sfpi::approx_recip(b_f);
    // One NR step: inv_b = inv_b * (2 - b * inv_b)
    sfpi::vFloat e = -inv_b_f * b_f + 1.0f;
    inv_b_f = e * inv_b_f + inv_b_f;

    // Initial quotient approximation: q = a * (1/b)
    sfpi::vFloat q_f = a_f * inv_b_f + vConstFloatPrgm0;
    sfpi::vUInt q = sfpi::exman(q_f);

    sfpi::vInt qb = sfpi::fractional_mul(q, b);
    qb <<= 10;

    // Compute initial remainder
    sfpi::vInt r = a - qb;

    // Compute correction for approximation error: correction = |r| / b
    sfpi::vFloat r_f = sfpi::convert<sfpi::vFloat>(sfpi::abs(r), sfpi::RoundMode::Nearest);
    sfpi::vMag correction = sfpi::convert<sfpi::vUInt16>(r_f * inv_b_f, sfpi::RoundMode::Nearest);

    // Compute correction * b (full 32-bit result from 24-bit multiplies)
    sfpi::vInt tmp_lo = sfpi::fractional_mul(correction, b);
    sfpi::vInt tmp_hi = sfpi::fractional_mul(correction, b, sfpi::FractionalHalf::High);
    sfpi::vInt b_hi = sfpi::fractional_mul(correction, b >> 23);
    sfpi::vInt tmp = tmp_lo + ((tmp_hi + b_hi) << 23);

    v_if(r < 0) { tmp = -tmp; }
    v_endif;
    r -= tmp;

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

// Unsigned (uint32) remainder. compute_unsigned_remainder_int32() is exact only when both
// operands are in [0, 2^31) (abs() is a no-op there), so we range-reduce into that regime:
// * b <  2^31: halve a to clear the problematic top bit. With t = a >> 1 (logical) and
//              a = 2*t + (a & 1), a % b = (2*(t % b) + (a & 1)) % b. t < 2^31 for every uint32 a,
//              so the single helper call always sees operands in [0, 2^31).
// * b >= 2^31: a < 2^32 <= 2*b, so a is already in [0, 2b) and needs no helper (a % b = a or a - b).
// Both regimes yield a value x in [0, 2b), reduced by one conditional subtract: x % b =
// (x >=u b) ? x - b : x. The SFPU integer compare only tests sign(x - b), which equals the true
// unsigned x >=u b except when b >= 2^31 and x < 2^31; a second predicate corrects those lanes
// (there x < b, so the remainder is x).
sfpi_inline void calculate_remainder_uint32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Load raw 32-bit patterns (interpreted as unsigned)
    sfpi::vInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // Call the helper unconditionally (nesting it inside predication crashes the SFPI rvtt_live
    // pass). t = (uint32)a >> 1 is always < 2^31, so the helper sees valid [0, 2^31) operands; rt
    // is only used on the b < 2^31 lanes, but every lane pays the call.
    sfpi::vInt t = sfpi::vInt(sfpi::vUInt(a) >> 1);
    sfpi::vInt rt = compute_unsigned_remainder_int32(t, b);

    // Reload a from DEST instead of keeping it live across the helper
    a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];

    // b < 2^31 uses x = 2*rt + (a & 1); b >= 2^31 keeps x = a
    v_if(b >= 0) { a = rt + rt + (a & 1); }
    v_endif;

    // x % b = (x >=u b) ? x - b : x, valid for both regimes since x in [0, 2b)
    sfpi::vInt r = a;
    v_if(sfpi::vUInt(a) >= sfpi::vUInt(b)) { r = a - b; }
    v_endif;
    // The above compare only tests sign(x - b), matching x >=u b except when b >= 2^31 and x < 2^31
    // Then x < b, remainder = x
    v_if(b < 0 && a >= 0) { r = a; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = r;
}

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_binary_remainder_(sfpi::vFloat in0, sfpi::vFloat in1) {
    // remainder(a, b) = a - floor(a/b) * b

    sfpi::vFloat a = in0;
    sfpi::vFloat b = in1;

    // Compute a/b = a * (1/b)
    sfpi::vFloat div_result = a * sfpu_reciprocal_iter<2>(b);

    // Compute floor(a/b)
    sfpi::vFloat floor_div = _floor_body_(div_result);

    // Compute remainder = a - floor(a/b) * b
    sfpi::vFloat result = a - floor_div * b;

    // Sign correction: remainder must match the sign of b (or be zero).
    // XOR of the float bit-patterns detects sign mismatch via the MSB,
    // avoiding a compound conditional with four comparisons and an OR.
    v_if(result != 0.0f) {
        sfpi::vInt signs = sfpi::as<sfpi::vInt>(result) ^ sfpi::as<sfpi::vInt>(b);
        v_and(signs < 0);
        result += b;
    }
    v_endif;

    // Magnitude correction: reciprocal imprecision can cause floor() to be greater than the true floor value.
    v_if(b > sfpi::vFloat(0.0f) && a > sfpi::vFloat(0.0f)) {
        sfpi::vFloat diff = result - b;
        v_if(diff >= sfpi::vFloat(0.0f)) { result = diff; }
        v_endif;
    }
    v_endif;
    v_if(b < sfpi::vFloat(0.0f) && a < sfpi::vFloat(0.0f)) {
        sfpi::vFloat diff = result - b;
        v_if(diff <= sfpi::vFloat(0.0f)) { result = diff; }
        v_endif;
    }
    v_endif;

    // Handle division by zero - return NaN
    v_if(b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
    v_endif;

    if constexpr (!is_fp32_dest_acc_en) {
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
    }

    return result;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_remainder_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_remainder_uint32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_uint32_body(dst_index_in0, dst_index_in1, dst_index_out);
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

template <bool APPROXIMATION_MODE>
inline void remainder_uint32_init() {
    // Shares the int32 setup: the unsigned path reuses compute_unsigned_remainder_int32().
    div_floor_init<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void remainder_binary_init() {
    sfpi::vConstFloatPrgm0 = 2.0f;
}
}  // namespace ckernel::sfpu
