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

// IEEE-754 +inf; abs bits == kInfBits is +-inf, abs bits > kInfBits is NaN.
constexpr int32_t kInfBits = 0x7F800000;

// Computes 1/|b| for the unsigned remainder (single Newton–Raphson refinement). Split recip and
// remainder computation so that the tensor-scalar path can hoist this loop-invariant work above its
// element loop, since a scalar divisor is identical for every lane and iteration.
// Tensor callers prepare the numerator in the reciprocal's dependency slots.
// The inlined callbacks keep this arithmetic shared with the scalar path without
// delaying numerator preparation until after the reciprocal. They must perform
// only independent numerator work and leave the reciprocal operands unchanged.
template <typename PrepareNumerator>
sfpi_inline sfpi::vFloat unsigned_remainder_recip_scheduled(const sfpi::vMag& b, PrepareNumerator prepare_numerator) {
    // Convert to float; handle 2^31 edge case where sign-magnitude conversion yields negative
    sfpi::vFloat b_f = sfpi::convert<sfpi::vFloat>(b, sfpi::RoundMode::Nearest);
    v_if(b_f < 0.0f) { b_f = TWO_POW_31; }
    v_endif;

    sfpi::vFloat inv_b_f = sfpi::approx_recip(b_f);
    // One NR step: inv_b = inv_b * (2 - b * inv_b)
    sfpi::vFloat e = -inv_b_f * b_f + 1.0f;
    // Fill the refinement MAD's dependency slot with numerator magnitude preparation.
    prepare_numerator();
    return e * inv_b_f + inv_b_f;
}

// Scalar callers hoist the reciprocal and have no per-row numerator to prepare here.
sfpi_inline sfpi::vFloat unsigned_remainder_recip(const sfpi::vMag& b) {
    return unsigned_remainder_recip_scheduled(b, []() {});
}

// Core remainder calculation with numerator magnitude, repaired numerator float,
// divisor magnitude, and reciprocal precomputed.
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
    sfpi::vMag a, sfpi::vFloat a_f, sfpi::vMag b, const sfpi::vFloat& inv_b_f) {
    // Initial quotient approximation: q = a * (1/b)
    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;
    sfpi::vUInt q = sfpi::exman(q_f);

    sfpi::vInt qb = sfpi::fractional_mul(q, b);
    qb <<= 10;

    // Compute initial remainder
    sfpi::vInt r = a - qb;

    // Compute correction for approximation error: correction = |r| / b.
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
    sfpi::vFloat correction_f = r_f * inv_b_f;
    // Fill the multiply's dependency slot with the independent divisor split.
    sfpi::vMag b_high = b >> 23;
    sfpi::vMag correction = sfpi::convert<sfpi::vUInt16>(correction_f, sfpi::RoundMode::Nearest);

    // Compute correction * b (full 32-bit result from 24-bit multiplies)
    // Issue the low product last to separate the high products from their sum.
    sfpi::vInt b_hi = sfpi::fractional_mul(correction, b_high);
    sfpi::vInt tmp_hi = sfpi::fractional_mul(correction, b, sfpi::FractionalHalf::High);
    sfpi::vInt tmp_lo = sfpi::fractional_mul(correction, b);
    sfpi::vInt tmp = tmp_lo + ((tmp_hi + b_hi) << 23);

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

    // Final adjustment to ensure r is in [0, b). The corrected remainder
    // cannot be INT_MIN.
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
    const sfpi::vInt& a_signed, sfpi::vMag b, const sfpi::vFloat& inv_b_f) {
    sfpi::vMag a = sfpi::abs(a_signed);
    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest);
    if constexpr (numerator_can_be_int_min) {
        v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
        v_endif;
    }
    return compute_unsigned_remainder_int32<numerator_can_be_int_min>(a, a_f, b, inv_b_f);
}

// Computes the unsigned remainder: |a| - floor(|a| / |b|) * |b|
// Returns: unsigned remainder r
template <bool numerator_can_be_int_min = true>
sfpi_inline sfpi::vInt compute_unsigned_remainder_int32(const sfpi::vInt& a_signed, const sfpi::vInt& b_signed) {
    sfpi::vMag b = sfpi::abs(b_signed);
    sfpi::vMag a;
    sfpi::vFloat inv_b_f = unsigned_remainder_recip_scheduled(b, [&]() { a = sfpi::abs(a_signed); });
    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest);
    if constexpr (numerator_can_be_int_min) {
        v_if(a_f < 0.0f) { a_f = TWO_POW_31; }
        v_endif;
    }
    return compute_unsigned_remainder_int32<numerator_can_be_int_min>(a, a_f, b, inv_b_f);
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

    // First form the truncating remainder, then adjust to the divisor's sign.
    sfpi::vInt sign = a_signed ^ b_signed;
    v_if(a_signed < 0) { r = -r; }
    v_endif;
    v_if(r != 0 && sign < 0) { r += b_signed; }
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
    sfpi::vInt rt = compute_unsigned_remainder_int32<false /* numerator_can_be_int_min */>(t, b);

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

// Adds b to a non-zero result whose sign differs from b's: a remainder is zero or carries the
// divisor's sign. XOR of the float bit patterns detects the sign mismatch via the MSB, avoiding a
// compound conditional with four comparisons and an OR.
sfpi_inline void correct_sign_to_divisor(sfpi::vFloat& result, const sfpi::vFloat& b) {
    v_if(result != sfpi::vFloat(0.0f)) {
        sfpi::vInt signs = sfpi::as<sfpi::vInt>(result) ^ sfpi::as<sfpi::vInt>(b);
        v_and(signs < 0);
        result += b;
    }
    v_endif;
}

// Infinite divisor: the reciprocal of +-inf is exactly 0, so the quotient * b term is 0 * inf =
// NaN. For finite a, remainder(a, +-inf) is a when a is zero or has the sign of b, else b
// (torch.remainder). Classify on the bit pattern, as ckernel_sfpu_isclose.h does, so a NaN divisor
// is excluded. An infinite or NaN a keeps the NaN: clear the sign with setsgn rather than
// sfpi::abs, because SFPABS leaves -NaN sign-set and it would read as a negative integer below
// kInfBits. b_abs is |b|, passed in so a loop-invariant divisor magnitude stays hoisted.
sfpi_inline void apply_remainder_inf_divisor(
    sfpi::vFloat& result, const sfpi::vFloat& a, const sfpi::vFloat& b, const sfpi::vFloat& b_abs) {
    sfpi::vInt a_magnitude_bits = sfpi::as<sfpi::vInt>(sfpi::setsgn(a, 0 /* sgn */));
    v_if(sfpi::as<sfpi::vInt>(b_abs) == kInfBits && a_magnitude_bits < kInfBits) {
        result = a;
        // Last statement of the block: narrow the lanes instead of nesting another v_if.
        v_and(a_magnitude_bits != 0);
        v_and((sfpi::as<sfpi::vInt>(a) ^ sfpi::as<sfpi::vInt>(b)) < 0);
        result = b;
    }
    v_endif;
}

// fmod(a, +-inf) is a for finite a and NaN for an infinite a (which an a == b special case would
// otherwise zero for inf/inf). Same classification as apply_remainder_inf_divisor.
sfpi_inline void apply_fmod_inf_divisor(sfpi::vFloat& result, const sfpi::vFloat& a, const sfpi::vFloat& b_abs) {
    v_if(sfpi::as<sfpi::vInt>(b_abs) == kInfBits) {
        result = a;
        v_and(sfpi::as<sfpi::vInt>(sfpi::setsgn(a, 0 /* sgn */)) >= kInfBits);
        result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN());
    }
    v_endif;
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
    correct_sign_to_divisor(result, b);

    // Magnitude correction: reciprocal imprecision can leave floor() one below the true floor, so
    // |result| >= |b|. After the sign correction result carries b's sign (or is zero), so the check
    // is the same for either sign of a; for an exact multiple with opposite signs the raw result is
    // b itself. Recompute the remainder from the quotient instead of subtracting b from result: for
    // an exact multiple (floor_div + 1) * b is a and the difference is exactly zero, whereas
    // result - b keeps the rounding of the earlier product and can land just inside |b|.
    // The b == +-0 lanes are overwritten with NaN below, whichever arm they take.
    sfpi::vFloat diff = a - (floor_div + sfpi::vFloat(1.0f)) * b;
    v_if(b < sfpi::vFloat(0.0f)) {
        v_if(diff <= sfpi::vFloat(0.0f)) { result = diff; }
        v_endif;
    }
    v_else {
        v_if(diff >= sfpi::vFloat(0.0f)) { result = diff; }
        v_endif;
    }
    v_endif;
    // A floor two above the true quotient leaves result opposite-signed after the single sign
    // correction above (or equal to -b for an exact multiple): step once more towards b's sign. A
    // correct remainder is zero or already carries b's sign, so this never fires on one.
    correct_sign_to_divisor(result, b);

    // Quotients of 2^24 and above: floor_div + 1 rounds to floor_div or floor_div + 2 in float, so
    // diff above is the raw result or a - (floor_div + 2) * b (which the sign test rejects unless
    // the floor was two low), and a floor one below the true quotient still leaves |result| >= |b|.
    // Subtract b if result has the sign of b, otherwise add b, so |result| shrinks; copysgn makes
    // that one subtraction. The step is gated on the quotient because at small quotients the
    // recompute above is exact and a correctly rounded remainder can legitimately equal b
    // (a = -1, b = 2^26), which must be kept.
    constexpr float kQuotientStepLimit = 16777216.0f;  // 2^24
    v_if(sfpi::abs(floor_div) >= sfpi::vFloat(kQuotientStepLimit) && sfpi::abs(result) >= sfpi::abs(b)) {
        result -= sfpi::copysgn(b, result);
    }
    v_endif;

    // Handle division by zero - return NaN
    v_if(b == sfpi::vFloat(0.0f)) { result = sfpi::vFloat(std::numeric_limits<float>::quiet_NaN()); }
    v_endif;

    apply_remainder_inf_divisor(result, a, b, sfpi::abs(b));

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
#pragma GCC unroll 2
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
#pragma GCC unroll 2
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
    sfpi::vConstFloatPrgm0 = 2.0f;
}
}  // namespace ckernel::sfpu
