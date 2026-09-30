// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included inside namespace sfpi. Exact selected recurrence and same-row suffix.
template <class Config, class Reciprocal, class Tail>
inline vFloat inverse_square_scalar(vFloat x, Reciprocal reciprocal, Tail tail) {
    vFloat zero = 0.0f;
    vFloat one = 1.0f;
    // psi1(x)=psi1(x+1)+1/x^2 for every positive x. Taking that one
    // recurrence step unconditionally makes the positive and reflection arms
    // share z=1+|x| and the same inverse-square pipeline.
    vFloat z = setsgn(x, 0) + one;
    constexpr float kRound = 0x1.8p23f;
    vFloat n = x + kRound;
    n = n - kRound;
    vFloat signed_one = copysgn(one, x);
    vFloat negative_mask = 0.5f - 0.5f * signed_one;
    vFloat q = x - negative_mask * n;
    vFloat u = reciprocal(z);
    // Same Bernoulli core, in Horner order: one fewer issued arithmetic op
    // and one less live temporary than the expanded u/u^2 reconstruction.
    vFloat core = ((Config::kP0 * u + 0.5f) * u + one) * u;
    vFloat iq = reciprocal(q);
    // Mask before squaring: on the positive arm the regularizer is dead, but
    // evaluating it at q=x would overflow for large finite BF16 inputs and
    // turn the final 0*regularizer into NaN.
    vFloat regularizer_q = negative_mask * q;
    vFloat q2 = regularizer_q * regularizer_q;
    vFloat regularizer = (Config::kP2[2] * q2 + Config::kP2[1]) * q2 + Config::kP2[0];
    vFloat result = iq * iq + copysgn(core, x);
    result = negative_mask * regularizer + result;
    v_if((x < zero) && (q == zero)) {
        result = std::numeric_limits<float>::infinity();
        if constexpr (Config::kWhRepair) {
            if constexpr (Config::kDirectTail) {
                result = tail(x);
            } else {
                v_if(x < Config::kFiniteThreshold) {
                    result = __builtin_bit_cast(float, uint32_t(Config::kFiniteRepresentative) << 16);
                }
                v_endif;
                v_if(x < -Config::kZeroTransitionPreviousMagnitude) { result = 0.0f; }
                v_endif;
            }
        }
    }
    v_endif;
    vInt source_exponent = exexp(setsgn(x, 0), ExponentMode::Biased);
    v_if(source_exponent == 255) { result = zero; }
    v_endif;
    if constexpr (!Config::kWhRepair) {
        v_if((source_exponent == 255) && (x < zero)) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    v_if(is_zero(x)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    v_if(x == 0x1p126f) { result = 0x1p-126f; }
    v_endif;
    return result;
}

#if defined(ARCH_BLACKHOLE)
template <class Config, unsigned Hold, unsigned Advance>
inline void inverse_square_replay() {
    constexpr unsigned kSharedInverseSquareBodySlots = 32;
    static constexpr unsigned kSharedInverseSquareFiniteThresholdBiasBf16 =
        __builtin_bit_cast(uint32_t, -Config::kFiniteThreshold) >> 16;
    static constexpr unsigned kSharedInverseSquareZeroTransitionBiasBf16 =
        __builtin_bit_cast(uint32_t, Config::kZeroTransitionPreviousMagnitude) >> 16;
    static constexpr unsigned kSharedInverseSquareNegInfResultBf16 = Config::kNegativeInfinityWord;
    constexpr uint16_t kRoundBiasBf16 = 0x4b40u;  // 1.5*2^23
    constexpr uint16_t kHalfBf16 = 0x3f00u;
    constexpr uint16_t kP0Bf16 = __builtin_bit_cast(uint32_t, Config::kP0) >> 16;
    constexpr uint16_t kP2TopBf16 = __builtin_bit_cast(uint32_t, Config::kP2[2]) >> 16;

    // Cross-replay pin.  Every body leaves L7 untouched.
    TTI_SFPLOADI(ckernel::p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_FLOATB, kHalfBf16);

    TTI_REPLAY(0, kSharedInverseSquareBodySlots, 1, 1);
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold,
                0);  // 1: raw decoded x
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                  1);  // 2: |x|
    TTI_SFPMOV(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG3,
               0);  // 3: preserve x sign carrier
    TTI_SFPSETSGN(0, ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG3,
                  0);  // 4: signed_one=copysign(1,x)
    TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB,
                 kRoundBiasBf16);  // 5: round bias
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        0);  // 6: x+bias
    TTI_SFPADD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        0);  // 7: z=1+|x|; also round-MAD gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        2);  // 8: n=(x+bias)-bias
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        1);  // 9: negative_mask=.5-.5*signed_one
    TTI_SFPARECIP(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG5,
                  0);  // 10: reciprocal(z) seed; also mask-MAD gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        1);      // 11: q=x-negative_mask*n
    TTI_SFPNOP;  // 12: q producer / reciprocal seed gap
    TTI_SFPARECIP(0, ckernel::p_sfpu::LREG2, ckernel::p_sfpu::LREG6,
                  0);  // 13: reciprocal(q) seed
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        2);  // 14: z*ru-2; also q-seed gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG0,
        2);  // 15: q*riq-2; also u-residual gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG5,
        3);  // 16: refined u=ru*(-residual)
    TTI_SFPSETCC(0, ckernel::p_sfpu::LREG0, 0,
                 0);  // 17: q residual < 0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG6,
        3);                    // 18: refined iq; predicate preserves q==0 +Inf seed
    TTI_SFPENCC(3, 0, 0, 10);  // 19
    TTI_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB,
                 kP0Bf16);  // 20: a0
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG1,
        0);  // 21: core1=a0*u+.5
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // 22: rq=negative_mask*q; core gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        0);  // 23: core2=core1*u+1
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        0);  // 24: q2=rq*rq; core gap
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG5,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG1,
        0);  // 25: core=core2*u
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG6,
        0);  // 26: iq2
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_FLOATB,
                 kP2TopBf16);  // 27: b2
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG13,
        ckernel::p_sfpu::LREG0,
        0);  // 28: reg1=b2*q2+b1
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG6,
        ckernel::p_sfpu::LREG1,
        0);  // 29: signed core + iq2; regularizer gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG14,
        ckernel::p_sfpu::LREG0,
        0);      // 30: regularizer=reg1*q2+b0
    TTI_SFPNOP;  // 31: final dual-MAD gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LREG2,
        0);  // 32: result=mask*regularizer+signed_core+iq2

    auto same_row_suffix = []() {
        // Round the finite replay result first.  Blackhole's FP32-to-BF16
        // stochastic-round instruction canonicalizes every NaN payload to
        // infinity, so typed raw-class replacements must happen after it.
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG2,
            ckernel::p_sfpu::LREG2,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // 1
        if constexpr (Config::kRepair) {
            // The selected form's typed terminal proves result=+Inf exactly on
            // finite nonpositive integers. Recompute that anonymous predicate
            // from the source, then use the target graph's abstractly derived
            // first-finite boundary. Finite-reference lanes are untouched.
            TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, Hold,
                        0);  // decoded source x
            TTI_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB,
                         0x4b40);  // 1.5*2^23 BF16 rounding bias
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG4,
                0);  // x+bias
            TTI_SFPNOP;
            TTI_SFPMAD(
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG1,
                ckernel::p_sfpu::LREG4,
                2);  // rint(x)=(x+bias)-bias
            TTI_SFPNOP;
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_neg1,
                ckernel::p_sfpu::LREG4,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG4,
                0);  // x-rint(x)
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, kSharedInverseSquareFiniteThresholdBiasBf16);
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG5,
                ckernel::p_sfpu::LREG5,
                0);  // x - graph-derived midpoint
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG4, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, 0);  // below midpoint
            TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kFiniteRepresentative);
            TTI_SFPENCC(0, 0, 0, 0);
            // At the graph-derived final transition every remaining negative BF16
            // finite value is integral. A strict comparison with the immediately
            // preceding representable magnitude therefore owns the complete
            // target +zero class without a numeric lookup table.
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG5, sfpi::SFPLOADI_MOD0_FLOATB, kSharedInverseSquareZeroTransitionBiasBf16);
            TTI_SFPADD(
                ckernel::p_sfpu::LCONST_1, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG5, ckernel::p_sfpu::LREG5, 0);
            TTI_SFPSETCC(0, ckernel::p_sfpu::LREG5, 0, 0);
            TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB, Config::kZeroRepresentative);
            TTI_SFPENCC(0, 0, 0, 0);
        }
        TTI_SFPLOAD(ckernel::p_sfpu::LREG0, sfpi::SFPLOAD_MOD0_FMT_UINT16, Hold,
                    0);  // 2: physical raw BF16
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG1,
            sfpi::SFPLOADI_MOD0_USHORT,
            0x00ff);  // 3: exponent-FF value and equality mask
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG1, ckernel::p_sfpu::LREG0,
                   0);  // 4: raw ^ 0x00ff, written to L0
        TTI_SFPAND(
            ckernel::p_sfpu::LREG1,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            1);  // 5: exponent==0xff iff result is zero
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0,
                     sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 6
        // The broad exponent-FF quotient canonicalizes first. The exact raw
        // -Inf terminal is restored below from the compiler's typed special
        // policy, rather than baking one activation/profile result here.
        TTI_SFPLOADI(ckernel::p_sfpu::LREG2, sfpi::SFPLOADI_MOD0_FLOATB,
                     0x0000);     // 7: every NaN and -Inf -> +0
        TTI_SFPENCC(0, 0, 0, 0);  // 8
        TTI_SFPLOAD(
            ckernel::p_sfpu::LREG0,
            sfpi::SFPLOAD_MOD0_FMT_UINT16,
            Hold,
            0);  // 9: physical raw BF16 for exact overrides
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT,
                     0x80ff);  // 10: exact -Inf physical word
        TTI_SFPXOR(
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG1,
            0);  // 11: +Inf delta; XOR destination is its third operand
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0,
                     sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 12
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_FLOATB,
            kSharedInverseSquareNegInfResultBf16);  // 13: typed exact -Inf result
        TTI_SFPENCC(0, 0, 0, 0);                    // 14
        TTI_SFPLOADI(ckernel::p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_USHORT,
                     0x00fd);  // 15: +2^126 physical word
        TTI_SFPXOR(0, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LREG1,
                   0);  // 16: boundary delta
        TTI_SFPSETCC(0, ckernel::p_sfpu::LREG1, 0,
                     sfpi::SFPSETCC_MOD1_LREG_EQ0);  // 17
        TTI_SFPLOADI(
            ckernel::p_sfpu::LREG2,
            sfpi::SFPLOADI_MOD0_FLOATB,
            Config::kRepair && !Config::kFiniteReferencePrecedence ? Config::kZeroRepresentative : 0x0080);

        TTI_SFPENCC(0, 0, 0, 0);  // 19
        TTI_SFPSTORE(ckernel::p_sfpu::LREG2, 0, Advance,
                     0);  // 20
    };
    same_row_suffix();
#pragma GCC unroll 32
    for (int d = 1; d < 32; d++) {
        TTI_REPLAY(0, kSharedInverseSquareBodySlots, 0, 0);
        same_row_suffix();
    }
    TTI_SETRWC(ckernel::p_setrwc::CLR_NONE, 0, 0, 0, 0, ckernel::p_setrwc::SET_D);
}
#endif
