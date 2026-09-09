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

// Computes the scaled reciprocal 1/|b| for the unsigned remainder (two Newton–Raphson iterations
// plus an exponent-based scale). Split recip and remainder computation so that the tensor-scalar
// path can hoist this loop-invariant work above its element loop, since a scalar divisor is identical
// for every lane and iteration.
// Tensor callers prepare the numerator in the reciprocal's dependency slots.
// The inlined callbacks keep this arithmetic shared with the scalar path without
// delaying numerator preparation until after the reciprocal. They must perform
// only independent numerator work and leave the reciprocal operands unchanged.
template <typename PrepareMagnitude, typename PrepareFloat>
sfpi_inline sfpi::vFloat unsigned_remainder_recip_scheduled(
    const sfpi::vInt& b_signed, PrepareMagnitude prepare_magnitude, PrepareFloat prepare_float) {
    // Get absolute value of b for reciprocal computation
    sfpi::vMag b = sfpi::abs(b_signed);

    // Convert to float for reciprocal computation
    // Handle edge case: if conversion results in negative
    sfpi::vFloat b_f = sfpi::convert<sfpi::vFloat>(b, sfpi::RoundMode::Nearest);
    v_if(b_f < 0.0f) { b_f = TWO_POW_31; }
    v_endif;

    // Compute reciprocal of b
    sfpi::vFloat neg_b_f = sfpi::copyman(-1.0f, b_f);

    sfpi::vFloat inv_b_f = sfpi::vConstFloatPrgm2 + sfpi::vConstFloatPrgm1 * neg_b_f;

    sfpi::vFloat scale = sfpi::setman(b_f, 0);

    // First Newton-Raphson iteration: inv_b_f = inv_b_f * (2 - inv_b_f * b_f)
    sfpi::vFloat t = inv_b_f * neg_b_f + 1.0f;
    scale = sfpi::as<sfpi::vFloat>((254 << 23) - sfpi::as<sfpi::vInt>(scale));
    inv_b_f = t * inv_b_f + inv_b_f;

    // Fill the first refinement MAD's dependency slot with the numerator magnitude.
    prepare_magnitude();
    // Second Newton-Raphson iteration
    sfpi::vFloat e = inv_b_f * neg_b_f + 1.0f;
    // Convert the numerator while the second refinement's error MAD completes.
    prepare_float();
    inv_b_f = e * inv_b_f + inv_b_f;

    // Apply scaling factor to finalize reciprocal
    return inv_b_f * scale;
}

// Scalar callers hoist the reciprocal and have no per-row numerator to prepare here.
sfpi_inline sfpi::vFloat unsigned_remainder_recip(const sfpi::vInt& b_signed) {
    return unsigned_remainder_recip_scheduled(b_signed, []() {}, []() {});
}

// Core remainder calculation with numerator magnitude, repaired numerator float,
// and the scaled reciprocal 1/|b| precomputed.
// Use 32-bit integer division from ckernel_sfpu_div_int32_floor.h
// Returns: unsigned remainder r
// All overloads share the numerator_can_be_int_min contract: false asserts that
// the numerator magnitude is strictly below 2^31 (as in the range-reduced UINT32
// callers). This skips the numerator's sign-magnitude repair and rules out a
// positive 2^31 residual. It does not rule out a negative INT_MIN residual from
// quotient overshoot, whose magnitude conversion must remain safe independently.
// The default true also supports the magnitude 2^31 from a signed INT32_MIN.
template <bool numerator_can_be_int_min = true>
sfpi_inline sfpi::vInt compute_unsigned_remainder_int32(
    sfpi::vMag a, sfpi::vFloat a_f, const sfpi::vInt& b_signed, const sfpi::vFloat& inv_b_f) {
    // Initial quotient approximation : q = a * 1/b
    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;
    // Fill the quotient MAD dependency slot with the divisor magnitude.
    sfpi::vMag b = sfpi::abs(b_signed);
    sfpi::vMag q = sfpi::exman(q_f);

    // 8388608.0f = 2^23 is used as a Bias for mantissa alignment
    sfpi::vFloat MANTISSA_ALIGNMENT_OFFSET = 8388608.0f;

    // Split q and b into 11-bit chunks to compute q * b
    // Shift out unwanted bits to avoid a mask register in this register-constrained helper.
    // Division and the small-divisor helper retain masks where they do not cause spills.
    constexpr unsigned INT32_BITS = 32;
    constexpr unsigned CHUNK_BITS = 11;
    constexpr unsigned HIGH_CHUNK_SHIFT = 2 * CHUNK_BITS;
    constexpr unsigned LOW_CHUNK_SHIFT = INT32_BITS - CHUNK_BITS;
    constexpr unsigned MID_CHUNK_SHIFT = INT32_BITS - HIGH_CHUNK_SHIFT;
    sfpi::vFloat q1 =
        sfpi::convert<sfpi::vFloat>(sfpi::vMag((q << LOW_CHUNK_SHIFT) >> LOW_CHUNK_SHIFT), sfpi::RoundMode::Nearest);
    sfpi::vFloat q2 = sfpi::convert<sfpi::vFloat>(q >> CHUNK_BITS, sfpi::RoundMode::Nearest);
    sfpi::vFloat b1 =
        sfpi::convert<sfpi::vFloat>(sfpi::vMag((b << MID_CHUNK_SHIFT) >> LOW_CHUNK_SHIFT), sfpi::RoundMode::Nearest);
    sfpi::vFloat b0 =
        sfpi::convert<sfpi::vFloat>(sfpi::vMag((b << LOW_CHUNK_SHIFT) >> LOW_CHUNK_SHIFT), sfpi::RoundMode::Nearest);

    // hi = q2 * b0 + q1 * b1 (high part)
    // lo = q1 * b0 (low part)
    // Interleave independent products and bias additions to avoid multiply-use NOPs.
    // Fresh stage values avoid inactive-lane dependencies from SFPI assignment,
    // which otherwise require extra registers when the caller is outlined by LTO.
    sfpi::vFloat hi = q2 * b0;
    sfpi::vFloat lo = q1 * b0;
    sfpi::vFloat hi_biased = hi + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat lo_biased = lo + MANTISSA_ALIGNMENT_OFFSET;
    sfpi::vFloat hi_sum = q1 * b1 + hi_biased;

    sfpi::vUInt qb = (sfpi::exman(lo_biased) << CHUNK_BITS) + (sfpi::exman(hi_sum) << HIGH_CHUNK_SHIFT);

    // Compute remainder from the retained numerator magnitude.
    sfpi::vInt r{a - qb};

    // abs(INT_MIN) remains INT_MIN, whose sign-magnitude conversion produces
    // -0.0 instead of the valid magnitude 2**31.
    // Keep this repair independent of the numerator bound: it also covers a
    // negative INT_MIN residual from an overshooting quotient approximation.
    // Do not drop the low bit with convert(abs(r) >> 1) + addexp: combined with
    // reciprocal error, the final adjustment can be insufficient. For example,
    // -2140947629 % -1 then returns -1 instead of 0 on Blackhole.
    sfpi::vFloat r_f = sfpi::convert<sfpi::vFloat>(sfpi::abs(r), sfpi::RoundMode::Nearest);
    v_if(r_f < 0.0f) { r_f = TWO_POW_31; }
    v_endif;

    // Compute correction: r / b in float32
    sfpi::vFloat correction_f = r_f * inv_b_f;
    // Fill the multiply's dependency slot with the independent high divisor chunk.
    sfpi::vFloat b2 = sfpi::convert<sfpi::vFloat>(b >> HIGH_CHUNK_SHIFT, sfpi::RoundMode::Nearest);
    auto correction = sfpi::convert<sfpi::vUInt16>(correction_f, sfpi::RoundMode::Nearest);
    correction_f = sfpi::convert<sfpi::vFloat>(correction, sfpi::RoundMode::Nearest);

    // Use fresh values so SFPI's predicated assignment does not retain the old
    // chunks across the residual calculation in an outlined/LTO-compiled caller.
    sfpi::vFloat correction_b0 =
        sfpi::convert<sfpi::vFloat>(sfpi::vMag((b << LOW_CHUNK_SHIFT) >> LOW_CHUNK_SHIFT), sfpi::RoundMode::Nearest);
    sfpi::vFloat correction_b1 =
        sfpi::convert<sfpi::vFloat>(sfpi::vMag((b << MID_CHUNK_SHIFT) >> LOW_CHUNK_SHIFT), sfpi::RoundMode::Nearest);

    // tmp = correction * (b2<<22 + b1<<11 + b0)
    // Issue the independent products before consuming them in the bias additions.
    sfpi::vFloat low = correction_f * correction_b0;
    sfpi::vFloat mid = correction_f * correction_b1;
    sfpi::vFloat top = correction_f * b2;
    low += MANTISSA_ALIGNMENT_OFFSET;
    mid += MANTISSA_ALIGNMENT_OFFSET;
    top += MANTISSA_ALIGNMENT_OFFSET;

    sfpi::vInt tmp{sfpi::exman(low) + (sfpi::exman(mid) << CHUNK_BITS) + (sfpi::exman(top) << HIGH_CHUNK_SHIFT)};
    // When q is zero, qb is also zero, so r=INT_MIN is the positive magnitude
    // 2**31. A negative residual with nonzero q instead needs a negative correction.
    if constexpr (numerator_can_be_int_min) {
        v_if(r < 0 && q != 0) { tmp = -tmp; }
        v_endif;
    } else {
        // A range-reduced unsigned numerator cannot produce the positive 2**31 residue.
        v_if(r < 0) { tmp = -tmp; }
        v_endif;
    }
    r -= tmp;

    // Final adjustment. The corrected remainder cannot be INT_MIN.
    // Reuse the subtraction for both the comparison and the adjusted result.
    sfpi::vInt r_minus_b = r - b;
    v_if(r < 0) { r += b; }
    v_elseif(r_minus_b >= 0) { r = r_minus_b; }
    v_endif;

    return r;
}

// Preserve the scalar entry point with a loop-invariant, precomputed reciprocal.
template <bool numerator_can_be_int_min = true>
sfpi_inline sfpi::vInt compute_unsigned_remainder_int32(
    const sfpi::vInt& a_signed, const sfpi::vInt& b_signed, const sfpi::vFloat& inv_b_f) {
    sfpi::vMag a = sfpi::abs(a_signed);
    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest);
    if constexpr (numerator_can_be_int_min) {
        v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
        v_endif;
    }
    return compute_unsigned_remainder_int32<numerator_can_be_int_min>(a, a_f, b_signed, inv_b_f);
}

// Computes the unsigned remainder: |a| - floor(|a| / |b|) * |b|
// Returns: unsigned remainder r
template <bool numerator_can_be_int_min = true>
sfpi_inline sfpi::vInt compute_unsigned_remainder_int32(const sfpi::vInt& a_signed, const sfpi::vInt& b_signed) {
    sfpi::vMag a;
    sfpi::vFloat a_f;
    sfpi::vFloat inv_b_f = unsigned_remainder_recip_scheduled(
        b_signed,
        [&]() { a = sfpi::abs(a_signed); },
        [&]() { a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest); });
    if constexpr (numerator_can_be_int_min) {
        v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
        v_endif;
    }
    return compute_unsigned_remainder_int32<numerator_can_be_int_min>(a, a_f, b_signed, inv_b_f);
}

// Signed (int32) remainder = a - floor(a / b) * b
sfpi_inline void calculate_remainder_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Load signed inputs
    sfpi::vInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
    sfpi::vInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // Initialize fresh values for the reloads: SFPI assignment preserves the old
    // value in inactive lanes, keeping both signed inputs live across the helper
    // and spilling registers when this function is outlined in a full LTO build.
    sfpi::vInt a_reloaded = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
    sfpi::vInt b_reloaded = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // First form the truncating remainder, then adjust to the divisor's sign.
    sfpi::vInt sign = a_reloaded ^ b_reloaded;
    v_if(a_reloaded < 0) { r = -r; }
    v_endif;
    v_if(r != 0 && sign < 0) { r += b_reloaded; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>() = r;
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
    // Size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Load raw 32-bit patterns (interpreted as unsigned)
    sfpi::vInt a = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
    sfpi::vInt b = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // Call the helper unconditionally (nesting it inside predication crashes the SFPI rvtt_live
    // pass). t = (uint32)a >> 1 is always < 2^31, so the helper sees valid [0, 2^31) operands; rt
    // is only used on the b < 2^31 lanes, but every lane pays the call.
    sfpi::vInt t = sfpi::vInt(sfpi::vUInt(a) >> 1);
    sfpi::vInt rt = compute_unsigned_remainder_int32<false /* numerator_can_be_int_min */>(t, b);

    // Fresh initialization avoids SFPI's inactive-lane dependency on the old
    // inputs, allowing them to die before the helper (including under LTO).
    sfpi::vInt a_reloaded = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
    sfpi::vInt b_reloaded = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();

    // b < 2^31 uses x = 2*rt + (a & 1); b >= 2^31 keeps x = a
    v_if(b_reloaded >= 0) { a_reloaded = rt + rt + (a_reloaded & 1); }
    v_endif;

    // x % b = (x >=u b) ? x - b : x, valid for both regimes since x in [0, 2b)
    sfpi::vInt r = a_reloaded;
    v_if(sfpi::vUInt(a_reloaded) >= sfpi::vUInt(b_reloaded)) { r = a_reloaded - b_reloaded; }
    v_endif;
    // The above compare only tests sign(x - b), matching x >=u b except when b >= 2^31 and x < 2^31
    // Then x < b, remainder = x
    v_if(b_reloaded < 0 && a_reloaded >= 0) { r = a_reloaded; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>() = r;
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
    v_if(result != sfpi::vFloat(0.0f)) {
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

// Force inlining so the scheduled reciprocal callbacks do not make SFPI outline
// this loop and lose constant tile indices at the caller.
template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_remainder_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

// Force inlining so the scheduled reciprocal callbacks do not make SFPI outline
// this loop and lose constant tile indices at the caller.
template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_remainder_uint32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_uint32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
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

template <bool APPROXIMATION_MODE>
inline void remainder_binary_init() {
    recip_init<APPROXIMATION_MODE, false, false>();
}

}  // namespace ckernel::sfpu
