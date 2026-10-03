// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Include inside namespace sfpi. Caller owns constants and tile finalization.
inline vFloat exp_root_rsqrt(vFloat x) {
    vInt i = as<vInt>(as<vUInt>(x) >> 1);
    vFloat y = as<vFloat>(vConstIntPrgm0 - i);
    vFloat xy = x * y;
    vFloat c = (-y) * xy;
    y = y * (vConstFloatPrgm1 + c * (vConstFloatPrgm2 + c));

    v_if(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    return y;
}

template <typename Config>
inline vFloat exp_root_eval(vFloat x) {
    static_assert(
        Config::kExpDegree == 2u && Config::kCoreDegree == 4u && Config::kCorrectionDegree == 1u &&
            Config::kCorrectionCoefficients[0] == 1.0f,
        "compact exp/root zone descriptor has unsupported leaf shape");

    vFloat magnitude = setsgn(x, 0);
    vFloat square = magnitude * magnitude;
    vFloat core = Config::kCoreCoefficients[Config::kCoreDegree];
#pragma GCC unroll 8
    for (int k = (int)Config::kCoreDegree - 1; k >= 0; --k) {
        core = core * square + Config::kCoreCoefficients[k];
    }
    if constexpr (Config::kOdd) {
        core = magnitude * core;
    }

    // Match the fitter's biased one-FMA exponent state exactly.  The fixed
    // shift moves the large coefficient scale into an exact exponent add.
    constexpr float kLog2E = 1.4426950408889634f;
    constexpr float kBiasedOffset = 127.0f - Config::kExponentShift * kLog2E;
    vFloat biased = magnitude * kLog2E + kBiasedOffset;
    vInt exponent = exexp(biased);
    vInt mantissa = exman(biased, MantissaMode::ImplicitOne);
    mantissa = shft(mantissa, exponent, ShiftMode::Logical);
    vFloat encoded = as<vFloat>(mantissa);
    vInt integer_part = exexp(encoded, ExponentMode::Biased);
    vMag fractional_magnitude = exman(encoded);
    vFloat fraction = convert<vFloat>(fractional_magnitude, RoundMode::Nearest) * 0x1p-23f;

    vFloat exp_polynomial = Config::kExpCoefficients[Config::kExpDegree];
#pragma GCC unroll 4
    for (int k = (int)Config::kExpDegree - 1; k >= 0; --k) {
        exp_polynomial = exp_polynomial * fraction + Config::kExpCoefficients[k];
    }
    vFloat root = exp_root_rsqrt(magnitude);
    vFloat inverse = root * root;
    vFloat scaled_root = exp_polynomial * root;
    vFloat correction_term = Config::kCorrectionCoefficients[1] * inverse;
    // c0 is canonically one, so the final scale multiply folds into this FMA.
    vFloat unscaled = correction_term * scaled_root + scaled_root;
    vInt unscaled_exponent = exexp(unscaled, ExponentMode::Biased);
    vFloat result = setexp(unscaled, integer_part + unscaled_exponent - 127);

    v_if(magnitude <= Config::kCoreBoundary) { result = core; }
    v_endif;
    v_if(magnitude >= Config::kFiniteTerminal) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    if constexpr (Config::kOdd) {
        result = copysgn(result, x);
    }
    if constexpr (Config::kOriginRepair) {
        // The joint fitter may certify one x*P(x^2) origin cell whose final FP32
        // product is flushed before destination BF16 RNE.  Compare its derived
        // normal-domain image, then build signed FP32 MIN_NORMAL by clearing the
        // original normal input's mantissa.  The certificate is bound to target
        // policy, the ordinary joint product, and exact c0 bytes; no activation
        // identity or raw-word fingerprint participates.
        vFloat origin_ftz_rne_key = addexp(magnitude, Config::kOriginShift);
        v_if(origin_ftz_rne_key == __builtin_bit_cast(float, Config::kOriginScaledBits)) { result = setman(x, 0); }
        v_endif;
    }
    return result;
}
