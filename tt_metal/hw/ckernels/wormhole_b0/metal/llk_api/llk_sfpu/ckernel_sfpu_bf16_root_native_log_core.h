// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Included inside sfpi. Exact selected total root/log recurrence/reflection.
//
// Every call reads and writes only its own tile: the input stays in its DEST row
// until the result replaces it, so it is reloaded where needed instead of being
// held across the two polynomial chains, and coefficients are immediates. The
// constants the two log evaluations share (ln 2 and the leading two mantissa
// coefficients) live in the programmable constant registers, which the caller's
// reciprocal leaves free.

template <class Config>
inline void root_native_log_constants() {
    static_assert(Config::kLogDegree >= 2);
    vConstFloatPrgm0 = 0.6931471805599453f;
    vConstFloatPrgm1 = Config::kLog[Config::kLogDegree];
    vConstFloatPrgm2 = Config::kLog[Config::kLogDegree - 1];
}

template <class Config, class Reciprocal>
inline __attribute__((always_inline)) vFloat root_native_log_eval(uint32_t row, Reciprocal reciprocal) {
    auto polynomial = [](vFloat coordinate, const float* coefficients, uint32_t leaf_degree) {
        vFloat value = coefficients[leaf_degree];
#pragma GCC unroll 16
        for (int index = static_cast<int>(leaf_degree) - 1; index >= 0; --index) {
            value = value * coordinate + coefficients[index];
        }
        return value;
    };
    auto normalized_log = [&](vFloat coordinate) {
        // The biased exponent is positive, so its conversion needs no sign handling;
        // removing the bias afterwards is exact.
        vInt exponent = exexp(coordinate, ExponentMode::Biased);
        vFloat mantissa = setexp(coordinate, 127);
        v_if(mantissa >= 1.5f) {
            mantissa = mantissa * 0.5f;
            exponent = exponent + 1;
        }
        v_endif;
        vFloat exponent_value = convert<vFloat>(as<vSMag>(exponent), RoundMode::Nearest) - 127.0f;
        vFloat value = vConstFloatPrgm1 * mantissa + vConstFloatPrgm2;
#pragma GCC unroll 16
        for (int index = static_cast<int>(Config::kLogDegree) - 2; index >= 0; --index) {
            value = value * mantissa + Config::kLog[index];
        }
        return exponent_value * vConstFloatPrgm0 + value;
    };

    // One structural coordinate owns all positive, recurrence, and reflected
    // consumers.  The BF16 lattice makes every negative |x|>=128 an integer
    // pole, so periodic reduction is needed only on the bounded remainder.
    vFloat x = dst_reg[row];
    vFloat z = setsgn(x, 0) + 1.0f;
    v_if(x >= 1.0f) { z = x; }
    v_endif;

    vFloat result;
    {
        // Finish normalization before opening the reciprocal/Horner chain.
        // Otherwise exponent, mantissa, reciprocal, and correction overlap
        // beyond the eight physical SFPU registers.
        vFloat log_z = normalized_log(z);
        vFloat inverse = reciprocal(z);
        vFloat correction = polynomial(inverse, Config::kCoefficients, Config::kDegree);
        static_assert(Config::kRoot == 2.0f, "the unit branch shares the core's root factor");
        vFloat root_factor = z - Config::kRoot;
        result = root_factor * (log_z + correction);
        v_if(root_factor < 0.0f) {
            vFloat unit_factor = (z - 1.0f) * root_factor;
            result = unit_factor * polynomial(z, Config::kUnit, Config::kUnitDegree);
        }
        v_endif;
    }

    vFloat fraction_magnitude;
    {
        vFloat bounded_negative = dst_reg[row];
        vFloat negative_lattice_bound = -128.0f;
        ordered_min_max(negative_lattice_bound, bounded_negative);
        vFloat zero_bound = 0.0f;
        ordered_min_max(bounded_negative, zero_bound);
        constexpr float kRound = 0x1.8p23f;
        vFloat nearest = bounded_negative + kRound;
        nearest = nearest - kRound;
        fraction_magnitude = setsgn(bounded_negative - nearest, 0);
    }
    vFloat log_sinc = polynomial(fraction_magnitude, Config::kSinc, Config::kSincDegree);
    // An integer x < 0 (zero fraction) and x = 0 are poles; their coordinate is not used.
    vFloat secondary_log_coordinate = dst_reg[row];
    v_if(secondary_log_coordinate < 0.0f) { secondary_log_coordinate = fraction_magnitude; }
    v_endif;
    v_if(secondary_log_coordinate >= 1.0f) { secondary_log_coordinate = 1.0f; }
    v_endif;
    vFloat secondary_log = normalized_log(secondary_log_coordinate);
    x = dst_reg[row];
    v_if(x < 0.0f) {
        result = -secondary_log - log_sinc - result;
        v_if(fraction_magnitude == 0.0f) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
    }
    v_elseif(x < 1.0f) { result = result - secondary_log; }
    v_endif;
    v_if(is_zero(x)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    return result;
}

// Callers retain source-owned finalization, the encoded-input terminals and the reciprocal.
template <class Config, class Finalize, class Encoded, class Reciprocal>
inline void root_native_log_tile(Finalize finalize, Encoded encoded, Reciprocal reciprocal) {
    root_native_log_constants<Config>();
    for (int d = 0; d < 32; d++) {
        vFloat y = root_native_log_eval<Config>(d, reciprocal);
        finalize(dst_reg[d], y);
        encoded(dst_reg[d].template mode<DataLayout::U16>(), y);
        if constexpr (Config::kBf16) {
            y = convert<vFloat16b>(y, RoundMode::Nearest);
        }
        dst_reg[d] = y;
    }
}
