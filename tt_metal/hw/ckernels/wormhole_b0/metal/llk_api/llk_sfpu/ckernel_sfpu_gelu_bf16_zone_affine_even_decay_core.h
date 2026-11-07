// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
namespace ckernel::sfpu::ttpoly::gelu {
using namespace ::sfpi;
using ::sfpi::DataLayout;
constexpr float zone_exp_fraction_scale(int power) {
    float scale = 1.0f;
    for (int index = 0; index < power; ++index) {
        scale *= 0x1p-23f;
    }
    return scale;
}

constexpr float kZoneNegativeHalfLog2E = -0.7213475204444817f;
constexpr bool kZonePrgmPreload = true;

// Preloaded reads the three constants from Prgm0..2 (zone_affine_even_decay_init); otherwise the
// call loads them itself.
template <class Config, bool Preloaded = kZonePrgmPreload>
__attribute__((always_inline)) inline vFloat zone_affine_even_decay_eval(vFloat x) {
    static_assert(
        Config::kExpDegree == 2u && Config::kCoreDegree == 5u && Config::kDecayDegree == 3u &&
            Config::kBodySlots == 25u && Config::kBodySlots <= Config::kReplayCapacity && Config::kPeakLive <= 8u,
        "zone-factored descriptor exceeds compact structural resources");

    vFloat coordinate = x;
    vFloat square = coordinate * coordinate;

    vFloat xlog2;
    if constexpr (Preloaded) {
        // Same SFPMAD; the multiplier operand reads the preloaded constant
        // register instead of a per-row two-issue SFPLOADI materialization.
        xlog2 = square * vConstFloatPrgm0 + 127.0f;
    } else {
        xlog2 = square * kZoneNegativeHalfLog2E + 127.0f;
    }
    vInt exponent = exexp(xlog2);
    vInt mantissa = exman(xlog2, MantissaMode::ImplicitOne);
    mantissa = shft(mantissa, exponent, ShiftMode::Logical);
    vFloat encoded = as<vFloat>(mantissa);
    vInt integer_part = exexp(encoded, ExponentMode::Biased);
    vMag fractional_magnitude = exman(encoded);
    vFloat fraction = convert<vFloat>(fractional_magnitude, RoundMode::Nearest);

    vFloat exp_polynomial;
    if constexpr (Preloaded) {
        exp_polynomial = vConstFloatPrgm2 * fraction + vConstFloatPrgm1;
        exp_polynomial = exp_polynomial * fraction + Config::kExpCoefficients[0] * zone_exp_fraction_scale(0);
    } else {
        exp_polynomial = Config::kExpCoefficients[Config::kExpDegree] * zone_exp_fraction_scale(Config::kExpDegree);
#pragma GCC unroll 4
        for (int k = (int)Config::kExpDegree - 1; k >= 0; --k) {
            exp_polynomial = exp_polynomial * fraction + Config::kExpCoefficients[k] * zone_exp_fraction_scale(k);
        }
    }
    vFloat decay_polynomial = Config::kDecayCoefficients[Config::kDecayDegree];
#pragma GCC unroll 4
    for (int k = (int)Config::kDecayDegree - 1; k >= 0; --k) {
        decay_polynomial = decay_polynomial * coordinate + Config::kDecayCoefficients[k];
    }
    vFloat unscaled = exp_polynomial * decay_polynomial;
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
__attribute__((always_inline)) inline void zone_affine_even_decay_origin_repair(
    vUInt raw_u16, vFloat x, vFloat& result) {
    static_assert(
        Config::kOriginSlope == 0.5f && Config::kOriginMask == 0x7fffu && Config::kOriginValue == 0x7f01u &&
            Config::kOriginMantissa == 0u && Config::kOriginPatterns == 2u,
        "zone-factored origin-half target repair lost its typed proof");
    vUInt delta = (raw_u16 ^ vUInt(Config::kOriginValue)) & vUInt(Config::kOriginMask);
    v_if(delta == 0u) { result = setman(x, Config::kOriginMantissa); }
    v_endif;
}

template <class Config, class Prepare, class Finalize, bool Preloaded = kZonePrgmPreload>
inline void zone_affine_even_decay_tile(Prepare prepare, Finalize finalize) {
#pragma GCC unroll 8
    for (int row = 0; row < 32; ++row) {
        vFloat raw = dst_reg[row];
        vFloat x = prepare(raw);
        vFloat result = zone_affine_even_decay_eval<Config, Preloaded>(x);
        zone_affine_even_decay_origin_repair<Config>(dst_reg[row].mode<DataLayout::U16>(), x, result);
        finalize(raw, result);
        result = convert<vFloat16b>(result, RoundMode::Nearest);
        dst_reg[row] = result;
    }
}

}  // namespace ckernel::sfpu::ttpoly::gelu
