// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Included inside sfpi. Exact selected total root/log recurrence/reflection.
template <class Config, uint32_t degree, class Reciprocal>
inline __attribute__((always_inline)) vFloat
root_native_log_eval(vFloat x, uint32_t handoff_row, Reciprocal reciprocal) {
    static_assert(degree == Config::kDegree);
    auto polynomial = [](vFloat coordinate, const float* coefficients, uint32_t leaf_degree) {
        const uint32_t base = coefficients == Config::kLog    ? Config::kLogBase
                              : coefficients == Config::kUnit ? Config::kUnitBase
                                                              : Config::kSincBase;
        vFloat value;
        if constexpr (Config::kStore) {
            value = dst_reg[base + leaf_degree].template mode<DataLayout::F32>();
        } else {
            value = coefficients[leaf_degree];
        }
#pragma GCC unroll 16
        for (int index = static_cast<int>(leaf_degree) - 1; index >= 0; --index) {
            if constexpr (Config::kStore) {
                vFloat c = dst_reg[base + index].template mode<DataLayout::F32>();
                value = value * coordinate + c;
            } else {
                value = value * coordinate + coefficients[index];
            }
        }
        return value;
    };
    auto normalized_log = [&](vFloat coordinate) {
        vInt exponent = exexp(coordinate, ExponentMode::Biased) - 127;
        vFloat mantissa = setexp(coordinate, 127);
        v_if(mantissa >= 1.5f) {
            mantissa = mantissa * 0.5f;
            exponent = exponent + 1;
        }
        v_endif;
        vInt exponent_magnitude = exponent;
        v_if(exponent < 0) { exponent_magnitude = ~exponent + 1; }
        v_endif;
        vFloat exponent_value = convert<vFloat>(as<vSMag>(exponent_magnitude), RoundMode::Nearest);
        v_if(exponent < 0) { exponent_value = -exponent_value; }
        v_endif;
        return exponent_value * 0.6931471805599453f + polynomial(mantissa, Config::kLog, Config::kLogDegree);
    };

    // One structural coordinate owns all positive, recurrence, and reflected
    // consumers.  The BF16 lattice makes every negative |x|>=128 an integer
    // pole, so periodic reduction is needed only on the bounded remainder.
    vFloat z = setsgn(x, 0) + 1.0f;
    v_if(x >= 1.0f) { z = x; }
    v_endif;

    {
        // Finish normalization before opening the reciprocal/Horner chain.
        // Otherwise exponent, mantissa, reciprocal, and correction overlap
        // beyond the eight physical SFPU registers.
        vFloat log_z = normalized_log(z);
        vFloat inverse = reciprocal(z);
        vFloat correction;
        if constexpr (Config::kStore) {
            correction = dst_reg[Config::kCoreBase + degree].template mode<DataLayout::F32>();
        } else {
            correction = Config::kCoefficients[degree];
        }
#pragma GCC unroll 8
        for (int index = degree - 1; index >= 0; --index) {
            if constexpr (Config::kStore) {
                vFloat c = dst_reg[Config::kCoreBase + index].template mode<DataLayout::F32>();
                correction = correction * inverse + c;
            } else {
                correction = correction * inverse + Config::kCoefficients[index];
            }
        }
        vFloat core_result = (z - Config::kRoot) * (log_z + correction);
        v_if(z < 2.0f) {
            vFloat unit_coordinate = z;
            vFloat unit_factor = (unit_coordinate - 1.0f) * (unit_coordinate - 2.0f);
            core_result = unit_factor * polynomial(unit_coordinate, Config::kUnit, Config::kUnitDegree);
        }
        v_endif;
        dst_reg[handoff_row].template mode<DataLayout::F32>() = core_result;
    }

    // Recreate the bounded fraction only after the positive core is complete;
    // keeping it live across the two polynomial chains spills the eight-LREG
    // SFPU.  This remains one element pass and one common evaluator.
    vFloat bounded_negative = x;
    vFloat negative_lattice_bound = -128.0f;
    ordered_min_max(negative_lattice_bound, bounded_negative);
    vFloat zero_bound = 0.0f;
    ordered_min_max(bounded_negative, zero_bound);
    constexpr float kRound = 0x1.8p23f;
    vFloat nearest = bounded_negative + kRound;
    nearest = nearest - kRound;
    vFloat fraction = bounded_negative - nearest;
    vFloat fraction_magnitude = setsgn(fraction, 0);
    vFloat secondary_log_coordinate = 1.0f;
    v_if((x < 0.0f) && (fraction_magnitude != 0.0f)) { secondary_log_coordinate = fraction_magnitude; }
    v_endif;
    v_if((x > 0.0f) && (x < 1.0f)) { secondary_log_coordinate = x; }
    v_endif;
    vFloat secondary_log = normalized_log(secondary_log_coordinate);
    vFloat log_sinc = polynomial(fraction_magnitude, Config::kSinc, Config::kSincDegree);
    vFloat result = dst_reg[handoff_row].template mode<DataLayout::F32>();
    v_if((x > 0.0f) && (x < 1.0f)) { result = result - secondary_log; }
    v_endif;
    v_if(x < 0.0f) { result = -secondary_log - log_sinc - result; }
    v_endif;
    v_if((x < 0.0f) && (fraction == 0.0f)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    v_if(is_zero(x)) { result = std::numeric_limits<float>::infinity(); }
    v_endif;
    return result;
}

// Callers retain source-owned preparation/finalization and reciprocal initialization.
template <class Config, class Prepare, class Finalize, class Encoded, class Reciprocal>
inline void root_native_log_tile(Prepare prepare, Finalize finalize, Encoded encoded, Reciprocal reciprocal) {
    if constexpr (Config::kStore) {
        static_assert(
            Config::kLogBase >= 64u && Config::kCoreBase + Config::kDegree + 1u <= 108u,
            "root coefficients overlap the data/handoff rows or exceed the isolated window");
#pragma GCC unroll 32
        for (uint32_t i = 0; i <= Config::kLogDegree; ++i) {
            dst_reg[Config::kLogBase + i].template mode<DataLayout::F32>() = vFloat(Config::kLog[i]);
        }
#pragma GCC unroll 32
        for (uint32_t i = 0; i <= Config::kUnitDegree; ++i) {
            dst_reg[Config::kUnitBase + i].template mode<DataLayout::F32>() = vFloat(Config::kUnit[i]);
        }
#pragma GCC unroll 32
        for (uint32_t i = 0; i <= Config::kSincDegree; ++i) {
            dst_reg[Config::kSincBase + i].template mode<DataLayout::F32>() = vFloat(Config::kSinc[i]);
        }
#pragma GCC unroll 32
        for (uint32_t i = 0; i <= Config::kDegree; ++i) {
            dst_reg[Config::kCoreBase + i].template mode<DataLayout::F32>() = vFloat(Config::kCoefficients[i]);
        }
    }
    auto evaluate_row = [&](int d) __attribute__((always_inline)) {
        if constexpr (Config::kRawShadow) {
            vUInt encoded_raw = dst_reg[d].template mode<DataLayout::U16>();
            dst_reg[Config::kRawShadowBase + d].template mode<DataLayout::U16>() = encoded_raw;
        }
        vFloat x_raw = dst_reg[d];
        vFloat x = prepare(x_raw);
        vFloat y = root_native_log_eval<Config, Config::kDegree>(x, d + 32, reciprocal);
        finalize(x_raw, y);
        if constexpr (Config::kSourceTerminal) {
            encoded(dst_reg[d].template mode<DataLayout::U16>(), y);
        }
        if constexpr (Config::kRawShadow) {
            vUInt terminal_raw = dst_reg[Config::kRawShadowBase + d].template mode<DataLayout::U16>();
            encoded(terminal_raw, y);
        }
        if constexpr (Config::kBf16) {
            y = convert<vFloat16b>(y, RoundMode::Nearest);
        }
        dst_reg[d] = y;
    };
    if constexpr (Config::kStore) {
#pragma GCC unroll 32
        for (int d = 0; d < 32; d++) {
            evaluate_row(d);
        }
    } else {
        for (int d = 0; d < 32; d++) {
            evaluate_row(d);
        }
    }
}
