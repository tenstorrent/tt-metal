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
        // q = min(x, 1 - x), taken from the computed complement so that it
        // cannot alias the DST-backed x. The form 0.5 - |x - 0.5| would lose
        // every x below the FP32 half-ULP of 0.5 (2^-25).
        vFloat complement = vFloat(1.0f) - x;
        vFloat q = complement;
        v_if(x < 0.5f) { q = x; }
        v_endif;

        vFloat m;
        vInt exponent;
        normalized_log_reduce(q, m, exponent);
        vFloat r = m - vFloat(1.0f);
        vFloat u = q - 0.5f;

        // Coefficients follow the two boundaries: C0, C1, then L0, L1, L2,
        // each polynomial from low to high degree.
        vFloat correction = sfpu_mad(lut[kCoeff + 1], u, lut[kCoeff + 0]);
        vFloat log_ratio = sfpu_mad(lut[kCoeff + 4], r, lut[kCoeff + 3]);
        log_ratio = sfpu_mad(log_ratio, r, lut[kCoeff + 2]);
        vFloat log_q = normalized_log_expand<0x3f317218u>(r * log_ratio, exponent);
        constexpr float kLn2 = 0.69314718055994530942f;
        vFloat log_one_minus_q = sfpu_mad(u, correction, -kLn2);
        // Reload x from DST for the classification below: keeping it live
        // across the log reduction exceeds Blackhole's eight LREGs.
        vFloat x_effective = prepare(dst_reg[d]);
        // The magnitude is positive on both halves of the unit interval; the
        // lower half takes the negative sign.
        vFloat result = log_one_minus_q - log_q;
        v_if(x_effective < 0.5f) { result = -result; }
        v_endif;

        // Inputs below 0 or from 1 up return +inf: logit(1) is +inf, and the
        // BF16 pack stores NaN as +inf. NaN inputs reach +inf through the log
        // arithmetic. Zero and subnormal inputs return -inf; they are found by
        // the raw BF16 exponent field because Blackhole can read a negative
        // subnormal as a nonzero value.
        v_if(x_effective < 0.0f) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
        v_if(x_effective >= 1.0f) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
        vUInt raw_u16 = dst_reg[d].mode<DataLayout::U16>();
        vUInt raw_exponent = raw_u16 & vUInt(0x00ffu);
        {
            // TT-NN's logit stores -inf for finite inputs of magnitude at least
            // kPositiveFirst; this kernel keeps that class, one interval of BF16
            // words mirrored by sign that ends where the exponent field is all ones.
            static_assert(
                Config::kPositiveEnd == 0x7f80u && Config::kNegativeFirst == (Config::kPositiveFirst | 0x8000u) &&
                Config::kNegativeEnd == (Config::kPositiveEnd | 0x8000u) &&
                Config::kPatternCount == 2u * (Config::kPositiveEnd - Config::kPositiveFirst));
            constexpr float target_phase_lower = __builtin_bit_cast(float, Config::kPositiveFirst << 16);
            vFloat magnitude = setsgn(x_effective, 0);
            v_if(magnitude >= target_phase_lower) {
                // Infinite and NaN inputs, just past the interval, keep +inf.
                v_if(raw_exponent != 0xffu) { result = -std::numeric_limits<float>::infinity(); }
                v_endif;
            }
            v_endif;
        }
        v_if(raw_exponent == 0u) { result = -std::numeric_limits<float>::infinity(); }
        v_endif;
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[d] = result;
    }
}
