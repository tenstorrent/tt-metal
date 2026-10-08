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
    // converted to float via sfpi::convert
    sfpi::vMag b = sfpi::abs(b_orig);

    // Convert to floats, but check for the edge case mentioned above.
    sfpi::vFloat b_f = sfpi::convert<sfpi::vFloat>(b, sfpi::RoundMode::Nearest);
    v_if(b_f < 0.0f) { b_f = 0x1.0p31f; }
    v_endif;

    // Compute 1/b accurate to ~22 bits of precision via Halley's Method.
    // Since the inputs can be as large as 2**31-1, this only gives us an
    // initial approximation.
    // We interleave SFPMAD with the loading and conversion of `a`.
    sfpi::vFloat inv_b_f = sfpi::approx_recip(b_f);
    sfpi::vFloat e = -inv_b_f * b_f + 1.0f;
    sfpi::vInt a_orig = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    e = e * e + e;
    sfpi::vMag a = sfpi::abs(a_orig);
    inv_b_f = e * inv_b_f + inv_b_f;
    sfpi::vFloat a_f = sfpi::convert<sfpi::vFloat>(a, sfpi::RoundMode::Nearest);
    v_if(a_f < 0.0f) { a_f = 0x1.0p31f; }
    v_endif;

    // Initial approximation q = a * 1/b.
    // We add a special mantissa alignment factor 2.0f**(23+10), which shifts
    // the mantissa so that we extract the top 22 bits of the result.
    sfpi::vFloat q_f = a_f * inv_b_f + sfpi::vConstFloatPrgm0;
    sfpi::vInt sign = a_orig ^ b_orig;
    sfpi::vMag q_m = sfpi::exman(q_f);

    // Compute qb = q * b.  This tells us how close our approximation `q` is to
    // the target `a`.  We split into 23-bit chunks.
    // Since inv_b is only accurate to ~22 bits, we only care about the upper
    // 22 bits, so we can compute qb = (q1<<10 + 0) * (b1<<22 + b0)
    //                               = (q1<<10) * b0

    sfpi::vInt qb{sfpi::fractional_mul(q_m, b)};
    // Fill the multiply's dependency slot with the independent quotient shift.
    sfpi::vInt q{q_m << 10};
    qb <<= 10;

    // Compute remainder.
    sfpi::vInt r = a - qb;
    // Shift before conversion so the valid magnitude 2**31 is representable as
    // a positive sign-magnitude integer. Dropping the low bit adds at most 1/|b|
    // to the correction error, on top of reciprocal and FP rounding error.
    // The single final adjustment relies on the ~22-bit reciprocal accuracy
    // from the Halley refinement above; do not weaken it without rechecking this
    // error budget, especially for odd residuals with |b| == 1.
    // Do not assume the same budget for ckernel_sfpu_binary_remainder.h: it uses
    // a less accurate reciprocal and retains the low bit (see its Blackhole
    // counterexample).
    sfpi::vFloat r_f = sfpi::convert<sfpi::vFloat>(sfpi::abs(r) >> 1, sfpi::RoundMode::Nearest);
    r_f = sfpi::addexp(r_f, 1 /* delta */);

    // Compute correction value in float32.
    sfpi::vFloat correction_f = r_f * inv_b_f;
    // Split b while the correction multiply completes, before consuming its result.
    sfpi::vMag b_high = b >> 23;
    sfpi::vMag correction = sfpi::convert<sfpi::vUInt16>(correction_f, sfpi::RoundMode::Nearest);

    // Compute tmp = correction * b.
    sfpi::vInt b1 = sfpi::fractional_mul(correction, b_high);
    sfpi::vInt tmp_hi = sfpi::fractional_mul(correction, b, sfpi::FractionalHalf::High);
    sfpi::vInt tmp_lo = sfpi::fractional_mul(correction, b);
    tmp_hi += b1;
    tmp_hi <<= 23;
    sfpi::vInt tmp = tmp_lo + tmp_hi;

    // Apply correction and adjust remainder.
    // When q is zero, qb is also zero, so r=INT_MIN represents the valid
    // positive magnitude 2**31 rather than a negative remainder.
    // Normalize the correction's sign so q and r can be updated unconditionally.
    sfpi::vInt cor = correction;
    v_if(r < 0 && q != 0) {
        tmp = -tmp;
        cor = -cor;
    }
    v_endif;
    // Keep this operand order with the sign updates above: it avoids an extra
    // move in current Blackhole SFPI allocation. Recheck both rounding modes.
    q = cor + q;
    r -= tmp;

    // Since the correction might have been rounded, we may need to correct one
    // additional bit.  The corrected remainder cannot be INT_MIN.
    // Reuse the subtraction for both the upper comparison and adjusted remainder.
    sfpi::vInt r_minus_b = r - b;
    v_if(r < 0) {
        q -= 1;
        r += b;
    }
    v_elseif(r_minus_b >= 0) {
        q += 1;
        r = r_minus_b;
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

#ifndef DISABLE_SFPLOADMACRO
sfpi_inline void div_int32_floor_lm_head(const uint in0, const uint in1) {
    // macro 0: SFPENCC
    TT_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG5 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG5 >> 2));
    TTI_SFPABS(0, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPABS_MOD1_INT);
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LCONST_0, 1);
    // dummy, bf < 0 lanes only; macro 1: e
    TT_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG0 >> 2));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 0)
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG1, sfpi::SFPARECIP_MOD1_RECIP);
    // macro 2: SFPENCC; with e = 1 - L1 * L0 (macro 1)
    TT_SFPLOADMACRO((2 << 2) | (p_sfpu::LREG2 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in0 | (p_sfpu::LREG2 >> 2));
    TTI_SFPABS(0, p_sfpu::LREG2, p_sfpu::LREG6, sfpi::SFPABS_MOD1_INT);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPCAST(p_sfpu::LREG6, p_sfpu::LREG3, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPGT(0, p_sfpu::LREG3, p_sfpu::LCONST_0, 1);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 2)
    TTI_SFPMAD(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LREG3, 0);
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG2, 0);
    TTI_SFPEXMAN(0, p_sfpu::LREG3, p_sfpu::LREG3, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPMUL24(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG5, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPSHFT(10, p_sfpu::LREG3, p_sfpu::LREG3, 5);
    TTI_SFPSHFT(10, p_sfpu::LREG5, p_sfpu::LREG5, 7);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPABS(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPSHFT(-1 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 5);
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPDIVP2(1, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSDIVP2_MOD1_ADD);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(-23 & 0xfff, p_sfpu::LREG4, p_sfpu::LREG7, 5);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, sfpi::SFPMUL24_MOD1_UPPER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
}

sfpi_inline void div_int32_floor_lm_tail(const uint out) {
    TTI_SFPSHFT(23, p_sfpu::LREG6, p_sfpu::LREG6, 7);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMOV(0, p_sfpu::LREG4, p_sfpu::LREG3, 2);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG3, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(-1 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG4, p_sfpu::LREG1, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPMOV(0, p_sfpu::LREG3, p_sfpu::LREG1, 0);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPIADD(-1 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, out);
    sfpi::dst_reg++;
}
#endif

template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_div_int32_floor(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_div_int32_body<true>(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
#else
    // SFPLOADMACRO schedule of calculate_div_int32_body<true>: 57 issues per row instead of 59; two
    // SFPENCCs share a cycle with the predicated SFPLOADI they close, which then still sees the old flags.
    const uint in0 = dst_index_in0 * 64, in1 = dst_index_in1 * 64, out = dst_index_out * 64;
    lltt::record<lltt::Exec>(0, 32);
    div_int32_floor_lm_head(in0, in1);
    div_int32_floor_lm_tail(out);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(0, 32);
        div_int32_floor_lm_tail(out);
    }
#endif
}

#ifndef DISABLE_SFPLOADMACRO
sfpi_inline void div_int32_trunc_lm_head(const uint in0, const uint in1) {
    // macro 0: SFPENCC
    TT_SFPLOADMACRO((0 << 2) | (p_sfpu::LREG5 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG5 >> 2));
    TTI_SFPABS(0, p_sfpu::LREG5, p_sfpu::LREG4, sfpi::SFPABS_MOD1_INT);
    TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPGT(0, p_sfpu::LREG0, p_sfpu::LCONST_0, 1);
    // dummy, bf < 0 lanes only; macro 1: e
    TT_SFPLOADMACRO((1 << 2) | (p_sfpu::LREG0 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in1 | (p_sfpu::LREG0 >> 2));
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 0)
    TTI_SFPARECIP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPARECIP_MOD1_RECIP);
    // macro 2: SFPENCC; with e = 1 - L2 * L0 (macro 1)
    TT_SFPLOADMACRO((2 << 2) | (p_sfpu::LREG3 & 3), InstrModLoadStore::INT32, ADDR_MOD_7, in0 | (p_sfpu::LREG3 >> 2));
    TTI_SFPABS(0, p_sfpu::LREG3, p_sfpu::LREG6, sfpi::SFPABS_MOD1_INT);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LREG0, 0);
    TTI_SFPCAST(p_sfpu::LREG6, p_sfpu::LREG1, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG2, p_sfpu::LREG2, 0);
    TTI_SFPGT(0, p_sfpu::LREG1, p_sfpu::LCONST_0, 1);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x4f00);  // with SFPENCC (macro 2)
    TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG12, p_sfpu::LREG1, 0);
    TTI_SFPXOR(0, p_sfpu::LREG5, p_sfpu::LREG3, 0);
    TTI_SFPEXMAN(0, p_sfpu::LREG1, p_sfpu::LREG1, sfpi::SFPEXMAN_MOD1_PAD9);
    TTI_SFPMUL24(p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG5, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPSHFT(10, p_sfpu::LREG1, p_sfpu::LREG1, 5);
    TTI_SFPSHFT(10, p_sfpu::LREG5, p_sfpu::LREG5, 7);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG5, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPABS(0, p_sfpu::LREG5, p_sfpu::LREG0, sfpi::SFPABS_MOD1_INT);
    TTI_SFPSHFT(-1 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, 5);
    TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPCAST_MOD1_SM32_TO_FP32_RNE);
    TTI_SFPDIVP2(1, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSDIVP2_MOD1_ADD);
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPSHFT(-23 & 0xfff, p_sfpu::LREG4, p_sfpu::LREG7, 5);
    TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPSTOCHRND_MOD1_FP32_TO_UINT16);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG7, p_sfpu::LCONST_0, p_sfpu::LREG7, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, sfpi::SFPMUL24_MOD1_UPPER);
    TTI_SFPMUL24(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG2, sfpi::SFPMUL24_MOD1_LOWER);
    TTI_SFPIADD(0, p_sfpu::LREG7, p_sfpu::LREG6, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
}

sfpi_inline void div_int32_trunc_lm_tail(const uint out) {
    TTI_SFPSHFT(23, p_sfpu::LREG6, p_sfpu::LREG6, 7);
    TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG5, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPSETCC(0, p_sfpu::LREG1, 0, sfpi::SFPSETCC_MOD1_LREG_NE0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG5, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG4, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(-1 & 0xfff, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPSETCC(0, p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    TTI_SFPIADD(1, p_sfpu::LREG0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TTI_SFPSETCC(0, p_sfpu::LREG3, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPIADD(
        0, p_sfpu::LCONST_0, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_NONE);
    TTI_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
    TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, out);
    sfpi::dst_reg++;
}
#endif

template <bool APPROXIMATION_MODE, int ITERATIONS>
sfpi_inline void calculate_div_int32_trunc(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_div_int32_body<false>(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
#else
    // SFPLOADMACRO schedule of calculate_div_int32_body<false>: 52 issues per row instead of 54; two
    // SFPENCCs share a cycle with the predicated SFPLOADI they close, which then still sees the old flags.
    const uint in0 = dst_index_in0 * 64, in1 = dst_index_in1 * 64, out = dst_index_out * 64;
    lltt::record<lltt::Exec>(0, 32);
    div_int32_trunc_lm_head(in0, in1);
    div_int32_trunc_lm_tail(out);
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(0, 32);
        div_int32_trunc_lm_tail(out);
    }
#endif
}

#ifndef DISABLE_SFPLOADMACRO
// Macros of calculate_div_int32_body; inv0 is in L1 for floor and in L2 for trunc.
template <bool floor>
inline void div_int32_macro_init() {
    // A disabled unit uses delay 7 so it cancels no pending instruction.
    constexpr std::uint32_t disabled = 7 << 3;
    constexpr std::uint32_t inv0 = floor ? p_sfpu::LREG1 : p_sfpu::LREG2;
    // InstructionTemplate[0]: SFPENCC as in the body.
    {
        constexpr std::uint32_t insn = TT_OP_SFPENCC(sfpi::SFPENCC_IMM12_BOTH, 0, 0, sfpi::SFPENCC_MOD1_EI_RI);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 0, 0);
    }
    // InstructionTemplate[1]: VD = -inv0 * VD + 1.0.
    {
        constexpr std::uint32_t insn = TT_OP_SFPMAD(inv0, 0, p_sfpu::LCONST_1, 0, 1);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, insn & 0xffff);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, insn >> 16);
        TTI_SFPCONFIG(0, 1, 0);
    }
    // Macro 0: SFPENCC with the first 2^31 fixup.
    {
        constexpr std::uint32_t simple_bits = (4 << 3) | (4 + 0);
        constexpr std::uint32_t mad_bits = disabled;
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    // Macro 1: e = 1 - inv0 * bf after the fixup.
    {
        constexpr std::uint32_t simple_bits = disabled;
        constexpr std::uint32_t mad_bits = 0x80 | (2 << 3) | (4 + 1);
        constexpr std::uint32_t round_bits = disabled;
        constexpr std::uint32_t store_bits = disabled;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    // Macro 2: SFPENCC with the second fixup.
    {
        constexpr std::uint32_t simple_bits = (5 << 3) | (4 + 0);
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
}
#endif

template <bool APPROXIMATION_MODE>
inline void div_trunc_init() {
    sfpi::vConstFloatPrgm0 = 8589934592.0f;
#ifndef DISABLE_SFPLOADMACRO
    div_int32_macro_init<false>();
#endif
}

// remainder and fmod inits call this too; each then programs its own macros.
template <bool APPROXIMATION_MODE>
inline void div_floor_init() {
    sfpi::vConstFloatPrgm0 = 8589934592.0f;
#ifndef DISABLE_SFPLOADMACRO
    div_int32_macro_init<true>();
#endif
}

}  // namespace ckernel::sfpu
