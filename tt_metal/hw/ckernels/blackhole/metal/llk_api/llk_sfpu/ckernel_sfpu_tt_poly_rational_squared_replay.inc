#if defined(EMBEDDED_LUT) && defined(USE_BF16) && defined(ARCH_BLACKHOLE) && \
    defined(SQUARED_ABS_DENOMINATOR_RATIONAL) &&                            \
    defined(TT_TARGET_BH_BF16_SQUARED_ABS_DENOMINATOR_TTI) &&               \
    !defined(RATIONAL_TTI_DISABLE) && !defined(TT_DOMAIN_ACTION_PROGRAM) && \
    !defined(PRECOMPOSE_INPUT_AFFINE) && !defined(POSTCOMPOSE_AFFINE_Y) &&  \
    !defined(POSTCOMPOSE_AFFINE_Y_TIMES_INPUT) &&                           \
    !defined(RATIONAL_RECIPROCAL_ZERO_ITER) &&                              \
    !defined(RATIONAL_RECIPROCAL_ONE_ITER)
#define SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE 1
constexpr bool kSquaredAbsDenominatorTtiReplay =
    (TT_TARGET_BH_BF16_SQUARED_ABS_DENOMINATOR_TTI_BODY_SLOTS == 18u) &&
    tti_replay_fits(TT_TARGET_BH_BF16_SQUARED_ABS_DENOMINATOR_TTI_BODY_SLOTS);

// Arm the existing BH load-macro RNE/store tail for this anonymous replay.
// T1 captures the same FP32->BF16 RNE used by the plain body; macro sequence
// zero fires it at issue+1 and the unchanged-format store at issue+3.  The
// template issue is backdoor-only; no programmable constant beyond the
// reciprocal helper's immutable 2.0 (Prgm0/L12) is consumed.
inline void squared_abs_denominator_tti_lm_init() {
    TTI_SFP_STOCH_RND(
        SFPSTOCHRND_RND_EVEN,
        0,
        0,
        ckernel::p_sfpu::LREG0,
        13,
        SFPSTOCHRND_MOD1_FP32_TO_FP16B);
    TTI_SFPLOADI(0, SFPLOADI_MOD0_LOWER, 0x0000);  // mad | simple idle
    TTI_SFPLOADI(0, SFPLOADI_MOD0_UPPER, 0x1385);  // store | round
    TTI_SFPCONFIG(0, 4, 0);
    TTI_SFPCONFIG(0x110, 8, 1);
}

inline void squared_abs_denominator_tti_replay_tile() {
    constexpr uint32_t kBodySlots = 18u;
    static_assert(kSquaredAbsDenominatorTtiReplay);
    TT_REPLAY_RECORD(kBodySlots);  // record + execute element 0
    TTI_SFPLOAD(ckernel::p_sfpu::LREG0, 0, TT_ADDRMOD_DST_HOLD, 0);  // x
    TTI_SFPNOP;  // conservative replay load-consumer gap
    TTI_SFPSETSGN(
        0,
        ckernel::p_sfpu::LREG0,
        ckernel::p_sfpu::LREG1,
        1);  // magnitude = |x| (no clamp: certificate-proved unnecessary)
    TTI_SFPMAD(
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_1,
        ckernel::p_sfpu::LREG7,
        0);  // den = magnitude + 1.0
    TTI_SFPNOP;  // den hazard gap
    TTI_SFPARECIP(0, 7, 2, 0);  // reciprocal seed -> L2
    TTI_SFPNOP;  // reciprocal-seed hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG3,
        2);  // t = den*r0 - 2.0
    TTI_SFPNOP;  // first-residual hazard gap
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG3,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG4,
        3);  // y1 = r0*-t - 0
    TTI_SFPNOP;
    TTI_SFPSETCC(0, 3, 0, 0);  // CC: first residual < 0 (NaN guard)
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG7,
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG12,
        ckernel::p_sfpu::LREG1,
        1);  // t2 = den*y1 - 2.0 (predicated)
    TTI_SFPNOP;
    TTI_SFPMAD(
        ckernel::p_sfpu::LREG4,
        ckernel::p_sfpu::LREG1,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG2,
        2);  // reciprocal = y1*-t2 - 0 (predicated)
    TTI_SFPENCC(3, 0, 0, 10);  // CC-balanced replay body
    TTI_SFPMUL(
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LREG2,
        ckernel::p_sfpu::LCONST_0,
        ckernel::p_sfpu::LREG0,
        0);  // y = reciprocal * reciprocal (the derivative's only extra op)
    // The load-macro itself is the result-MUL hazard gap.  Captured T1 RNE
    // fires at issue+1, then the fixed store advances the same destination
    // row at issue+3. L4 is dead after the second Newton update.
    TTI_SFPLOADMACRO(
        (0 << 2) | (ckernel::p_sfpu::LREG4 & 3),
        0,
        TT_ADDRMOD_DST_ADV,
        (ckernel::p_sfpu::LREG4 >> 2));
    TT_REPLAY_REST(kBodySlots, 31);  // elements 1..31
    TT_REPLAY_TILE_EPILOGUE();
}
#else
#define SQUARED_ABS_DENOMINATOR_TTI_CANDIDATE 0
constexpr bool kSquaredAbsDenominatorTtiReplay = false;
#endif
