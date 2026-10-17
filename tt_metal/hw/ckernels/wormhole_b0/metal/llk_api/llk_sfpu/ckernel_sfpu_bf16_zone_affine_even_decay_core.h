// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace sfpi {
// Closed anonymous three-leaf form.  The body evaluates one bounded P2 exp2
// leaf, one P5 affine-even core, and one P3 negative decay correction.  The
// decay correction is multiplied before exponent scaling so a normal final
// result is never lost through a standalone subnormal exp intermediate.
constexpr float zone_exp_fraction_scale(int power) {
    float scale = 1.0f;
    for (int index = 0; index < power; ++index) {
        scale *= 0x1p-23f;
    }
    return scale;
}

// Zone constant-pool preload (programmed ONCE in kernel_main; see the HW_PRELOAD
// precedent there): Prgm0 carries the biased-exponent multiplier and Prgm1/2 the
// two upper pre-scaled exponent-leaf coefficients, so the per-row body issues no
// SFPLOADI pair for any of the three.  The programmable constant registers sit
// outside the eight-LREG budget, so the body's peak live set is unchanged.
// TT_ZONE_PRGM_PRELOAD_DISABLE (via TT_ACT_KERNEL_EXTRA_DEFINES) is the A/B
// escape arm: the identical arithmetic with in-body literal materialization.
constexpr float kZoneNegativeHalfLog2E = -0.7213475204444817f;

template <class Config>
inline vFloat zone_affine_even_decay_eval(vFloat x) {
    static_assert(
        Config::kExpDegree == 2u && Config::kCoreDegree == 5u && Config::kDecayDegree == 3u &&
            Config::kBodySlots == 25u && Config::kBodySlots <= Config::kReplayCapacity && Config::kPeakLive <= 8u,
        "zone-factored descriptor exceeds compact structural resources");

    // The terminal and identity predicates below own every lane outside the
    // fitted core/decay coordinates.  Evaluate the raw coordinate speculatively
    // and overwrite those lanes, matching the established SFPU tail topology;
    // clamping here only duplicated the already-total route ownership.
    vFloat coordinate = x;
    vFloat square = coordinate * coordinate;

    // Moroz-style biased exponent decomposition matching the fitter model.
#if !defined(TT_ZONE_PRGM_PRELOAD_DISABLE)
    // Same SFPMAD; the multiplier operand reads the preloaded constant
    // register instead of a per-row two-issue SFPLOADI materialization.
    vFloat xlog2 = square * vConstFloatPrgm0 + 127.0f;
#else
    vFloat xlog2 = square * kZoneNegativeHalfLog2E + 127.0f;
#endif
    vInt exponent = exexp(xlog2);
    vInt mantissa = exman(xlog2, MantissaMode::ImplicitOne);
    mantissa = shft(mantissa, exponent, ShiftMode::Logical);
    vFloat encoded = as<vFloat>(mantissa);
    vInt integer_part = exexp(encoded, ExponentMode::Biased);
    vMag fractional_magnitude = exman(encoded);
    vFloat fraction = convert<vFloat>(fractional_magnitude, RoundMode::Nearest);

    // Fold the exact exman 2^-23 normalization into each semantic leaf
    // coefficient.  The CSV remains in its ordinary [0,1) basis while the
    // physical Horner consumes the raw integer magnitude and saves one SFPMUL.
#if !defined(TT_ZONE_PRGM_PRELOAD_DISABLE)
    // The P2 Horner over the preloaded upper coefficients: the head fuses the
    // init and first step into the one SFPMAD they always were
    // (c2'*fraction + c1'), with both constant operands read from Prgm2/Prgm1.
    // The FMA sequence, operand values, and rounding are identical to the
    // escape arm below; only the operand sourcing changes.
    vFloat exp_polynomial = vConstFloatPrgm2 * fraction + vConstFloatPrgm1;
    exp_polynomial = exp_polynomial * fraction + Config::kExpCoefficients[0] * zone_exp_fraction_scale(0);
#else
    vFloat exp_polynomial = Config::kExpCoefficients[Config::kExpDegree] * zone_exp_fraction_scale(Config::kExpDegree);
#pragma GCC unroll 4
    for (int k = (int)Config::kExpDegree - 1; k >= 0; --k) {
        exp_polynomial = exp_polynomial * fraction + Config::kExpCoefficients[k] * zone_exp_fraction_scale(k);
    }
#endif
    vFloat decay_polynomial = Config::kDecayCoefficients[Config::kDecayDegree];
#pragma GCC unroll 4
    for (int k = (int)Config::kDecayDegree - 1; k >= 0; --k) {
        decay_polynomial = decay_polynomial * coordinate + Config::kDecayCoefficients[k];
    }
    vFloat unscaled = exp_polynomial * decay_polynomial;
    // Synthesize the exact 2^(integer_part-127) factor from its biased exponent
    // bits.  Multiplication preserves unscaled's own exponent automatically and
    // lets BH FTZ the final product, replacing the five-op corrected-setexp
    // sequence with the established SHFT+SFPMUL pair.
    vFloat result = unscaled * as<vFloat>(integer_part << 23);

    v_if(x > Config::kDecayCoreBoundary) {
        vFloat core_polynomial = Config::kCoreCoefficients[Config::kCoreDegree];
#pragma GCC unroll 8
        for (int k = (int)Config::kCoreDegree - 1; k >= 0; --k) {
            core_polynomial = core_polynomial * square + Config::kCoreCoefficients[k];
        }
        vFloat cdf = coordinate * core_polynomial + Config::kOriginSlope;
        result = coordinate * cdf;
    }
    v_endif;

    // x * 0 is 0 below the terminal, and NaN for -Inf or a NaN. WH SFPMAD gives
    // that NaN the sign of x times the zero's, so -0 makes it positive there.
    v_if(x <= Config::kNegativeTerminal) { result = x * -0.0f; }
    v_endif;
    v_if(x > Config::kPositiveIdentity) { result = x; }
    v_endif;
    return result;
}

template <class Config>
inline void zone_affine_even_decay_origin_repair(vUInt raw_u16, vFloat x, vFloat& result) {
    static_assert(
        Config::kOriginSlope == 0.5f && Config::kOriginMask == 0x7fffu && Config::kOriginValue == 0x7f01u &&
            Config::kOriginMantissa == 0u && Config::kOriginPatterns == 2u,
        "zone-factored origin-half target repair lost its typed proof");
    // Blackhole U16 DST layout is sign15|mantissa14:8|exponent7:0.  Masking
    // only sign therefore recognizes exactly IEEE BF16 0x00ff and 0x80ff.
    // SETMAN(0) keeps the decoded input sign/exponent and constructs signed
    // FP32 MIN_NORMAL before the ordinary BF16 RNE/store.  This is the target
    // rounding of the declared half-origin reconstruction, not a witness table.
#if !defined(TT_ZONE_ORIGIN_HALF_DISABLE)
    vUInt delta = (raw_u16 ^ vUInt(Config::kOriginValue)) & vUInt(Config::kOriginMask);
#else
    // C-VER escape arm: two literal physical-word deltas implementing the
    // same typed target result.  It is intentionally less compact but remains
    // output-byte-identical to the folded sign-agnostic predicate.
    constexpr uint16_t positive_word = Config::kOriginValue;
    constexpr uint16_t negative_word = Config::kOriginValue | (uint16_t)(~Config::kOriginMask);
    vUInt positive_delta = raw_u16 ^ vUInt(positive_word);
    vUInt negative_delta = raw_u16 ^ vUInt(negative_word);
    vUInt delta = positive_delta & negative_delta;
#endif
    v_if(delta == 0u) { result = setman(x, Config::kOriginMantissa); }
    v_endif;
}

template <class Config>
inline void zone_affine_even_decay_init() {
    vConstFloatPrgm0 = kZoneNegativeHalfLog2E;
    vConstFloatPrgm1 = Config::kExpCoefficients[1] * zone_exp_fraction_scale(1);
    vConstFloatPrgm2 = Config::kExpCoefficients[2] * zone_exp_fraction_scale(2);
}

template <class Config, class Prepare, class Finalize>
inline void zone_affine_even_decay_tile(Prepare prepare, Finalize finalize) {
#pragma GCC unroll 8
    for (int row = 0; row < 32; ++row) {
        vFloat raw = dst_reg[row];
        vFloat x = prepare(raw);
        vFloat result = zone_affine_even_decay_eval<Config>(x);
        zone_affine_even_decay_origin_repair<Config>(dst_reg[row].mode<DataLayout::U16>(), x, result);
        finalize(raw, result);
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[row] = result;
    }
}

}  // namespace sfpi
