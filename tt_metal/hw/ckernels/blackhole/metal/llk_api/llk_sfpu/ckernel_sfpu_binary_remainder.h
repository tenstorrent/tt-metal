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

#ifndef DISABLE_SFPLOADMACRO
sfpi_inline void remainder_int32_lm_head(const uint in0, const uint in1) {
    // macro 0: SFPENCC
    TT_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG1 >> 2));
    TT_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::INT32, ADDR_MOD_7, in0);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPABS_MOD1_INT);
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LCONST_0, 1);
    // dummy, bf < 0 lanes only; macro 1: e, SFPENCC
    TT_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG0 >> 2));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 0)
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG6, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG7, sfpi::SFPABS_MOD1_INT);  // with e = 1 - L6 * L0 (macro 1)
    TTI_SFPCAST(p_sfpu::LREG7, p_sfpu::LREG5, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LREG6, p_sfpu::LREG6, 0);
    TTI_SFPGT(0, p_sfpu::LREG5, p_sfpu::LCONST_0, 1);
    TTI_SFPLOADI(p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 1)
    TTI_SFPMAD(p_sfpu::LREG5, p_sfpu::LREG6, p_sfpu::LREG12, p_sfpu::LREG5, 0);
    // dummy; macro 2: SFPENCC
    TT_SFPLOADMACRO((2 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG0 >> 2));
    TTI_SFPEXMAN(0, p_sfpu::LREG5, p_sfpu::LREG5, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPMUL24(p_sfpu::LREG5, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPSHFT(10, p_sfpu::LREG4, p_sfpu::LREG4, 7);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPABS(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LCONST_0, 1);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 2)
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG6, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(-23 & 0xfff, p_sfpu::LREG3, p_sfpu::LREG7, 5);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG6, sfpi::SFPMUL24_MOD1_UPPER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT(23, p_sfpu::LREG6, p_sfpu::LREG6, 7);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
}

sfpi_inline void remainder_int32_lm_tail(const uint out) {
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG4, 2);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 2);
    TTI_SFPXOR(0, p_sfpu::LREG1, p_sfpu::LREG3, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, out);
    sfpi::dst_reg++;
}
#endif

// Force inlining so the scheduled reciprocal callbacks do not make SFPI outline
// this loop and lose constant tile indices at the caller.
template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_remainder_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
#else
    // SFPLOADMACRO schedule of calculate_remainder_int32_body: 55 issues per row instead of 57; three
    // SFPENCCs share a cycle with the predicated SFPLOADI they close, which then still sees the old flags.
    const uint in0 = dst_index_in0 * 64, in1 = dst_index_in1 * 64, out = dst_index_out * 64;
    lltt::record<lltt::Exec>(0, 32);
    remainder_int32_lm_head(in0, in1);
    remainder_int32_lm_tail(out);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(0, 32);
        remainder_int32_lm_tail(out);
    }
#endif
}

#ifndef DISABLE_SFPLOADMACRO
sfpi_inline void remainder_uint32_lm_head(const uint in0, const uint in1) {
    // macro 0: a >>= 1
    TT_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in0 | (p_sfpu::LREG0 >> 2));
    // macro 1: SFPENCC
    TT_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG1 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG1 >> 2));
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG2, sfpi::SFPABS_MOD1_INT);
    TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG3, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG3, p_sfpu::LCONST_0, 1);
    // dummy, bf < 0 lanes only; macro 2: e
    TT_SFPLOADMACRO((2 << 2) | (p_sfpu::LREG3 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG3 >> 2));
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 1)
    TTI_SFPARECIP(0, p_sfpu::LREG3, p_sfpu::LREG4, sfpi::SFPARECIP_MOD1_RECIP);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);  // with e = 1 - L4 * L3 (macro 2)
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG6, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG4, p_sfpu::LREG4, 0);
    TTI_SFPSHFT(-23 & 0xfff, p_sfpu::LREG2, p_sfpu::LREG5, 5);
    TTI_SFPMAD(p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG12, p_sfpu::LREG6, 0);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 2);
    TTI_SFPEXMAN(0, p_sfpu::LREG6, p_sfpu::LREG6, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPMUL24(p_sfpu::LREG6, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG6, sfpi::SFPMUL24_MOD1_LOWER);
    // dummy; macro 3: SFPENCC
    TT_SFPLOADMACRO((3 << 2) | (p_sfpu::LREG7 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG7 >> 2));
    TTI_SFPSHFT(10, p_sfpu::LREG6, p_sfpu::LREG6, 7);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPABS(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LCONST_0, 1);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 3)
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TT_SFPLOAD(p_sfpu::LREG7, InstrModLoadStore::INT32, ADDR_MOD_7, in0);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG5, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG4, sfpi::SFPMUL24_MOD1_UPPER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSHFT(23, p_sfpu::LREG4, p_sfpu::LREG4, 7);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
}

sfpi_inline void remainder_uint32_lm_tail(const uint out) {
    TTI_SFPSETCC(0, p_sfpu::LREG6, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_USHORT, 0x0001);
    TTI_SFPAND(p_sfpu::LREG7, p_sfpu::LREG3, p_sfpu::LREG3, 1);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG7, 0);
    TTI_SFPMOV(0, p_sfpu::LREG7, p_sfpu::LREG0, 2);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG2, 2);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 2);
    TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG2, 0);
    TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 2);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TT_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, out);
    sfpi::dst_reg++;
}
#endif

// Force inlining so the scheduled reciprocal callbacks do not make SFPI outline
// this loop and lose constant tile indices at the caller.
template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_remainder_uint32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_remainder_uint32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
#else
    // SFPLOADMACRO schedule of calculate_remainder_uint32_body: 63 issues per row instead of 65; two
    // SFPENCCs share a cycle with the predicated SFPLOADI they close, which then still sees the old flags.
    const uint in0 = dst_index_in0 * 64, in1 = dst_index_in1 * 64, out = dst_index_out * 64;
    lltt::record<lltt::Exec>(0, 32);
    remainder_uint32_lm_head(in0, in1);
    remainder_uint32_lm_tail(out);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(0, 32);
        remainder_uint32_lm_tail(out);
    }
#endif
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
#ifndef DISABLE_SFPLOADMACRO
    // A disabled unit uses delay 7 so it cancels no pending instruction.
    constexpr std::uint32_t disabled = 7 << 3;
    // InstructionTemplate[0]: SFPENCC as in the body.
    {
        constexpr std::uint32_t insn = TT_OP_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 0, 0);
    }
    // InstructionTemplate[1]: VD = -L6 * VD + 1.0.
    {
        constexpr std::uint32_t insn = TT_OP_SFPMAD(p_sfpu::LREG6, 0, p_sfpu::LCONST_1, 0, 1);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 1, 0);
    }
    // Macro 0: SFPENCC with the first 2^31 fixup.
    {
        constexpr std::uint32_t simple_bits = (5 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Macro 1: e = 1 - L6 * bf after the fixup, SFPENCC with the second fixup.
    {
        constexpr std::uint32_t simple_bits = (6 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = 0x80 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    // Macro 2: SFPENCC with the third fixup.
    {
        constexpr std::uint32_t simple_bits = (7 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // Misc: every unit counts issued instructions; no scheduled store.
    TTI_SFPCONFIG(0xf00, 8, 1);
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE>
inline void remainder_uint32_init() {
    // Shares the int32 setup: the unsigned path reuses compute_unsigned_remainder_int32().
    div_floor_init<APPROXIMATION_MODE>();
#ifndef DISABLE_SFPLOADMACRO
    // A disabled unit uses delay 7 so it cancels no pending instruction.
    constexpr std::uint32_t disabled = 7 << 3;
    // InstructionTemplate[0]: t = a >> 1 in place.
    {
        constexpr std::uint32_t insn = TT_OP_SFPSHFT(-1 & 0xfff, 0, 0, 5);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 0, 0);
    }
    // InstructionTemplate[1]: SFPENCC as in the body.
    {
        constexpr std::uint32_t insn = TT_OP_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 1, 0);
    }
    // InstructionTemplate[2]: VD = -L4 * VD + 1.0.
    {
        constexpr std::uint32_t insn = TT_OP_SFPMAD(p_sfpu::LREG4, 0, p_sfpu::LCONST_1, 0, 1);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 2, 0);
    }
    // Macro 0: t = a >> 1 in place, next cycle.
    {
        constexpr std::uint32_t simple_bits = (0 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Macro 1: SFPENCC with the first 2^31 fixup.
    {
        constexpr std::uint32_t simple_bits = (4 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    // Macro 2: e = 1 - L4 * bf after the fixup.
    {
        constexpr std::uint32_t simple_bits = disabled;
        constexpr std::uint32_t mad_bits = 0x80 | (2 << 3) | (4 + 2);
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    // Macro 3: SFPENCC with the second fixup.
    {
        constexpr std::uint32_t simple_bits = (5 << 3) | (4 + 1);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }
    // Misc: every unit counts issued instructions; no scheduled store.
    TTI_SFPCONFIG(0xf00, 8, 1);
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool APPROXIMATION_MODE>
inline void remainder_binary_init() {
    sfpi::vConstFloatPrgm0 = 2.0f;
}
}  // namespace ckernel::sfpu
