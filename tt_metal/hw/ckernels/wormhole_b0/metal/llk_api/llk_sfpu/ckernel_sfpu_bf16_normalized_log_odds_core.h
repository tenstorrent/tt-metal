// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Include inside namespace sfpi; callers retain ingress and LUT ownership.
inline void normalized_log_reduce(vFloat x, vFloat& m, vInt& e_int) {
    // Extract biased exponent: for x = 2^e * m, biased_exp = e + 127
    vInt biased_exp = exexp(x, ExponentMode::Biased);

    // Unbias: e = biased_exp - 127
    // But we want m in [1, 2), so we set exponent to 127 (bias)
    // This gives m = x / 2^e = x * 2^(-e) = x with exponent = 127
    e_int = biased_exp - 127;

    // Normalize mantissa to [1, 2) by setting exponent bits to 127
    m = setexp(x, 127);
}

// Reconstruct: log(x) = e*ln(2) + log(m)
// poly_result = log(m) where m ∈ [1, 2)
// Final result = e*ln(2) + poly_result
template <uint32_t ExpandBits>
inline vFloat normalized_log_expand(vFloat poly_result, vInt e_int) {
    // LOG_EXPAND_CONSTANT is base-specific: ln(2) for log, 1.0 for log2, log10(2) for log10
    constexpr float EXPAND_C = __builtin_bit_cast(float, ExpandBits);
    // int32_to_float expects SIGN-MAGNITUDE format, not two's complement.
    // Negative exponents must be converted: twos complement → sign-magnitude.
    v_if(e_int < 0) { e_int = as<vInt>(setsgn(as<vSMag>(~e_int + 1), 1)); }
    v_endif;
    vFloat e_float = convert<vFloat>(as<vSMag>(e_int), RoundMode::Nearest);
    return e_float * EXPAND_C + poly_result;
}

template <typename Config, typename Prepare, size_t LUT_SIZE>
inline void normalized_log_odds_tile(const std::array<float, LUT_SIZE>& lut, Prepare prepare) {
    static_assert(Config::kCorrectionDegree == 1, "first qualified separable correction is P1");
    static_assert(Config::kLogDegree == 2, "first qualified separable log ratio is P2");
    static_assert(LUT_SIZE == 7, "unit-interval boundaries plus five flattened coefficients");
    constexpr uint32_t kCoeff = 2;
#pragma GCC unroll 8
    for (int d = 0; d < 32; d++) {
        vFloat x = prepare(dst_reg[d]);
        // q=min(x,1-x), with q born from the computed complement so it cannot
        // alias the DST-backed x carrier.  The tempting centered identity
        // 0.5-abs(x-0.5) loses x below the fp32 half-ULP of 0.5 (2^-25), so
        // the predicate is required to preserve every bottom-binade normal.
        vFloat complement = vFloat(1.0f) - x;
        vFloat q = complement;
        v_if(x < 0.5f) { q = x; }
        v_endif;

        vFloat m;
        vInt exponent;
        normalized_log_reduce(q, m, exponent);
        vFloat r = m - vFloat(1.0f);
        vFloat u = q - 0.5f;

        // CSV layout: C0,C1,L0,L1,L2.  Both chains are ordinary low-to-high
        // fitted coefficients; no activation identity participates.
        vFloat correction = sfpu_mad(lut[kCoeff + 1], u, lut[kCoeff + 0]);
        vFloat log_ratio = sfpu_mad(lut[kCoeff + 4], r, lut[kCoeff + 3]);
        log_ratio = sfpu_mad(log_ratio, r, lut[kCoeff + 2]);
        vFloat log_q = normalized_log_expand<0x3f317218u>(r * log_ratio, exponent);
        constexpr float kLn2 = 0.69314718055994530942f;
        vFloat log_one_minus_q = sfpu_mad(u, correction, -kLn2);
        // Reconstruction is complete.  Reload the decoded and encoded source
        // coordinates from DST now, after the arithmetic temporaries retire.
        // Keeping either classifier live across log reduction exceeds the
        // eight-LREG Blackhole budget and corrupts endpoint/raw predicates.
        vFloat x_effective = prepare(dst_reg[d]);
        // The magnitude is positive on both halves of the unit interval.
        // Restore the lower-half sign with the freshly reloaded coordinate;
        // this keeps the sign carrier out of reduction/Horner liveness while
        // preserving the exact anonymous basis semantics.
        vFloat result = log_one_minus_q - log_q;
        v_if(x_effective < 0.5f) { result = -result; }
        v_endif;

        // Close the unit interval in the decoded coordinate. NaNs already
        // flow through the log arithmetic to +Inf. The remaining encoded
        // distinction shared by both profiles is exponent-zero: Blackhole's raw
        // BF16 ingress can expose negative subnormal payloads as nonzero SFPU
        // values even though the declared input policy is DAZ.  A late masked
        // equality closes that whole zero/subnormal class without a raw sign,
        // mantissa, NaN, or exterior classifier.
        v_if(x_effective < 0.0f) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
        v_if(x_effective >= 1.0f) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        vUInt raw_exponent = raw_u16 & vUInt(0x00ffu);
        if constexpr (Config::kIntervalCount != 0) {
            // S55 exhaustively evaluates the declared rounded target graph and
            // proves that its differing exterior class is one sign-mirrored raw
            // interval ending at exponent-FF. S60 emits those derived endpoints;
            // this anonymous lowering consumes the quotient, not the source graph.
            static_assert(Config::kIntervalCount == 2u);
            static_assert(
                Config::kPositiveEnd == 0x7f80u && Config::kNegativeFirst == (Config::kPositiveFirst | 0x8000u) &&
                Config::kNegativeEnd == (Config::kPositiveEnd | 0x8000u) &&
                Config::kPatternCount == 2u * (Config::kPositiveEnd - Config::kPositiveFirst));
            constexpr float target_phase_lower = __builtin_bit_cast(float, Config::kPositiveFirst << 16);
            vFloat magnitude = setsgn(x_effective, 0);
            v_if(magnitude >= target_phase_lower) {
                // Exponent-FF begins exactly at the interval's exclusive end.
                // Reject it in encoded space so NaN/Inf retain the declared +Inf.
                v_if(raw_exponent != 0xffu) { result = -std::numeric_limits<float>::infinity(); }
                v_endif;
            }
            v_endif;
        }
        v_if(raw_exponent == 0u) { result = -std::numeric_limits<float>::infinity(); }
        v_endif;
        if constexpr (Config::kBf16) {
            result = convert<vFloat16b>(result, RoundMode::Nearest);
        }
        dst_reg[d] = result;
    }
}
