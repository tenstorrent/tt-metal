// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

static_assert(
    TT_SELECTED_STIRLING_AGGREGATE_COUNT == 2u || TT_SELECTED_STIRLING_AGGREGATE_COUNT == 4u,
    "aggregate Stirling schedule supports the declared fixed-parameter family");

inline vFloat selected_stirling_log_ratio(vFloat x) {
    vFloat y = TT_SELECTED_STIRLING_LOG_RATIO[10];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[9];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[8];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[7];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[6];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[5];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[4];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[3];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[2];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[1];
    y = y * x + TT_SELECTED_STIRLING_LOG_RATIO[0];
    return y;
}

inline vFloat selected_stirling_correction(vFloat x) {
    vFloat y = TT_SELECTED_STIRLING_CORRECTION[12];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[11];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[10];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[9];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[8];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[7];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[6];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[5];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[4];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[3];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[2];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[1];
    y = y * x + TT_SELECTED_STIRLING_CORRECTION[0];
    return y;
}

// Natural log of a positive normal argument from its mantissa in [0.75, 1.5).
inline vFloat selected_stirling_positive_log(vFloat a) {
    vInt exponent = exexp(a, ExponentMode::Biased);
    vFloat mantissa = setexp(a, 127);
    v_if(mantissa >= 1.5f) {
        mantissa = mantissa * 0.5f;
        exponent = exponent + 1;
    }
    v_endif;
    vFloat residual = mantissa - 1.0f;
    vFloat log_mantissa = residual * selected_stirling_log_ratio(residual);
    vFloat exponent_float = convert<vFloat>(convert<vSMag>(exponent - 127), RoundMode::Nearest);
    return log_mantissa + exponent_float * 0.6931471805599453f;
}

// The aggregate Stirling form at u >= the correction coordinate scale.
inline vFloat selected_stirling_tail(vFloat u) {
    constexpr float count = static_cast<float>(TT_SELECTED_STIRLING_AGGREGATE_COUNT);
    constexpr float log_offset = __builtin_bit_cast(float, TT_SELECTED_STIRLING_LOG_OFFSET_BITS);
    constexpr float constant = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CONSTANT_BITS);
    constexpr float correction_scale = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORRECTION_SCALE_BITS);
    vFloat log_u = selected_stirling_positive_log(u);
    vFloat count_log = log_u * count;
    vFloat count_log_minus_count = count_log - count;
    vFloat scaled_main = u * count_log_minus_count;
    vFloat offset_log = log_u * log_offset;
    vFloat without_log_offset = scaled_main - offset_log;
    vFloat stirling_main = without_log_offset + constant;
    vFloat inverse_u = ckernel::sfpu::sfpu_reciprocal_iter<2>(u);
    vFloat correction_coordinate = inverse_u * correction_scale;
    return stirling_main + selected_stirling_correction(correction_coordinate);
}

inline vFloat selected_stirling_log_sine_ratio(vFloat w) {
    constexpr int last = TT_SELECTED_STIRLING_LOG_SINE_RATIO.size() - 1;
    vFloat y = TT_SELECTED_STIRLING_LOG_SINE_RATIO[last];
#pragma GCC unroll 16
    for (int i = last - 1; i >= 0; i--) {
        y = y * w + TT_SELECTED_STIRLING_LOG_SINE_RATIO[i];
    }
    return y;
}

// Below the core, the lgamma terms are reflected together:
//   F(x) = K - log(sin^2(2 pi x)) - F(origin - x),
// and F at t = origin - x below the correction scale is shifted up by the recurrence
//   F(t) = F(t + n) - log|prod_{k < n, j < p} (t + k - j/2)|,
// so the Stirling form runs at u >= scale. With r = 2x - round(2x) in [-1/2, 1/2],
//   log(sin^2(2 pi x)) = log(r^2) + log(sin^2(pi r) / r^2),
// so one logarithm covers both singular parts:
//   F(x) = K + s log 2 - log(sin^2(pi r) / r^2) + log(|prod| 2^-s / r^2) - Stirling(u).
// A pole is r == 0, or a magnitude at which every float is an integer.
inline void apply_selected_core_aggregate_stirling(vFloat x, vFloat& y) {
    constexpr float core_lower = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_LOWER_BITS);
    constexpr float core_upper = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_UPPER_BITS);
    constexpr float overflow = __builtin_bit_cast(float, TT_SELECTED_STIRLING_OVERFLOW_BITS);
    constexpr float origin = __builtin_bit_cast(float, TT_SELECTED_STIRLING_REFLECTION_ORIGIN_BITS);
    constexpr float reflection_constant = __builtin_bit_cast(float, TT_SELECTED_STIRLING_REFLECTION_CONSTANT_BITS);
    constexpr float shift_bound = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORRECTION_SCALE_BITS);
    constexpr float shift = static_cast<float>(TT_SELECTED_STIRLING_REFLECTION_SHIFT);
    constexpr float log_scale = __builtin_bit_cast(float, TT_SELECTED_STIRLING_REFLECTION_LOG_SCALE_BITS);
    // 1.5 * 2^23: adding and subtracting it rounds |v| < 2^22 to an integer.
    constexpr float round_bias = 12582912.0f;

    // Below the core, y holds the reflection's terms other than Stirling(u).
    v_if(x <= core_lower) {
        vFloat t = origin - x;
        vFloat product = 1.0f;
        v_if(t < shift_bound) {
            vFloat repeated = t + TT_SELECTED_STIRLING_REFLECTION_SQUARED_OFFSETS[0];
#pragma GCC unroll 16
            for (uint32_t i = 1; i < TT_SELECTED_STIRLING_REFLECTION_SQUARED_OFFSETS.size(); i++) {
                repeated = repeated * (t + TT_SELECTED_STIRLING_REFLECTION_SQUARED_OFFSETS[i]);
            }
            product = repeated * repeated;
#pragma GCC unroll 16
            for (uint32_t i = 0; i < TT_SELECTED_STIRLING_REFLECTION_SINGLE_OFFSETS.size(); i++) {
                product = product * (t + TT_SELECTED_STIRLING_REFLECTION_SINGLE_OFFSETS[i]);
            }
        }
        v_endif;
        vFloat twice = x * 2.0f;
        vFloat nearest = twice + round_bias;
        nearest = nearest - round_bias;
        vFloat r = twice - nearest;
        vFloat square = r * r;
        y = selected_stirling_log_sine_ratio(square);
        y = reflection_constant - y;
        // At a pole the logarithm below is finite or garbage, and inf absorbs it.
        v_if(square == 0.0f) { y = std::numeric_limits<float>::infinity(); }
        v_endif;
        vFloat argument = setsgn(product, 0);
        if constexpr (log_scale != 1.0f) { argument = argument * log_scale; }
        argument = argument * ckernel::sfpu::sfpu_reciprocal_iter<2>(square);
        y = y + selected_stirling_positive_log(argument);
    }
    v_endif;

    // One Stirling evaluation serves the tail above the core and the reflection below it.
    v_if(x > core_upper || x <= core_lower) {
        vFloat u = x;
        v_if(x <= core_lower) {
            u = origin - x;
            v_if(u < shift_bound) { u = u + shift; }
            v_endif;
        }
        v_endif;
        vFloat stirling = selected_stirling_tail(u);
        v_if(x <= core_lower) { y = y - stirling; }
        v_else { y = stirling; }
        v_endif;
    }
    v_endif;

    // 2^22: every float of at least this magnitude is an integer.
    v_if(is_nan(x)) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x >= overflow) { y = std::numeric_limits<float>::infinity(); }
    v_elseif(x <= -4194304.0f) { y = std::numeric_limits<float>::infinity(); }
    // The replaced composite subtracts the pole-producing shift in BF16, which lands on that
    // pole, +Inf, on this open window.
    v_elseif(
        x > __builtin_bit_cast(float, TT_SELECTED_STIRLING_REFLECTION_STOCK_WINDOW_BITS[0]) &&
        x < __builtin_bit_cast(float, TT_SELECTED_STIRLING_REFLECTION_STOCK_WINDOW_BITS[1])) {
        y = std::numeric_limits<float>::infinity();
    }
    v_endif;
}

inline vFloat selected_aggregate_stirling_prepare(vFloat x_raw) {
    constexpr float core_lower = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_LOWER_BITS);
    constexpr float core_upper = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_UPPER_BITS);
    vFloat prepared = x_raw;
    v_if(prepared < core_lower) { prepared = core_lower; }
    v_elseif(prepared > core_upper) { prepared = core_upper; }
    v_endif;
    return prepared;
}
