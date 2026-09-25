#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR)

// Inline Cody-Waite exp(arg) — no overflow/underflow checks needed because
// the asymptotic region has known bounded input range.
// Degree-5 Taylor for exp(r), |r| < ln(2)/2: relative error < 3.3e-9.
inline vFloat asymptotic_exp(vFloat arg) {
    constexpr float INV_LN2 = 1.4426950408889634f;
    vFloat z = arg * INV_LN2;

    // Round to nearest integer (branch-free magic-number technique)
    const vFloat c231 = ckernel::sfpu::Converter::as_float(0x4B400000U);
    vFloat tmp = z + c231;
    vFloat k = tmp - c231;
    vInt k_int = as<vInt>(tmp) - as<vInt>(c231);

    // Cody-Waite extended precision: r = arg - k*ln2
    constexpr float NEG_LN2_HI = -0.6931152343750000f;
    constexpr float NEG_LN2_LO = -3.19461832987e-05f;
    vFloat r = k * NEG_LN2_HI + arg;
    r = k * NEG_LN2_LO + r;

    // Degree-5 Taylor for exp(r)
    vFloat p = 1.0f / 120.0f;
    p = p * r + 1.0f / 24.0f;
    p = p * r + 1.0f / 6.0f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;

    // Scale by 2^k via exponent bit manipulation.
    //
    // UNDERFLOW GUARD: setexp writes the exponent FIELD directly, so a biased
    // exponent that lands at or below 0 wraps into a large positive exponent
    // and returns ~1e38 where the true value has underflowed to ~0.  Measured
    // on Blackhole via erf_bw = (2/sqrt(pi))*exp(-x^2): the wrap begins exactly
    // where round(-x^2/ln2) < -127, i.e. |x| >= 9.4375, which returned 2.72e38
    // against a golden of 2.35e-39 (150 grid points, symmetric in +/-x).  The
    // clamp must come AFTER setexp: setexp(0.0f, e) rebuilds a nonzero value
    // from the zero mantissa, so zeroing p beforehand does not help.
    vInt p_exp = exexp(p, ExponentMode::Biased);
    vInt new_exp = p_exp + k_int;
    vFloat out = setexp(p, new_exp);
    v_if(new_exp <= 0) { out = 0.0f; }
    v_endif;
    return out;
}

#endif  // ASYMPTOTIC_FACTOR_EXP_*
