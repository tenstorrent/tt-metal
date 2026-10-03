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

inline void apply_selected_core_aggregate_stirling(vFloat x, vFloat& y) {
    constexpr float core_upper = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_UPPER_BITS);
    constexpr float overflow = __builtin_bit_cast(float, TT_SELECTED_STIRLING_OVERFLOW_BITS);
    constexpr float count = static_cast<float>(TT_SELECTED_STIRLING_AGGREGATE_COUNT);
    constexpr float log_offset = __builtin_bit_cast(float, TT_SELECTED_STIRLING_LOG_OFFSET_BITS);
    constexpr float constant = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CONSTANT_BITS);
    constexpr float correction_scale = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORRECTION_SCALE_BITS);

    v_if(x > core_upper && x < overflow) {
        vFloat magnitude = setsgn(x, 0);
        vInt biased_exponent = exexp(magnitude, ExponentMode::Biased);
        vFloat mantissa = setexp(magnitude, 127);
        vInt exponent = biased_exponent - 126;
        v_if(mantissa < 1.5f) { exponent = biased_exponent - 127; }
        v_endif;
        vSMag exponent_smag = convert<vSMag>(exponent);
        vFloat exponent_float = convert<vFloat>(exponent_smag, RoundMode::Nearest);
        vInt negative_exponent = -exponent;
        vFloat normalized = x;
        vInt normalized_exponent = exexp(x, ExponentMode::Biased) + negative_exponent;
        v_if(is_nan(x)) { normalized = x; }
        v_elseif(exexp(x, ExponentMode::Biased) == 255) { normalized = x; }
        v_elseif(is_zero(x)) { normalized = x; }
        v_elseif(normalized_exponent >= 255) {
            normalized = copysgn(vFloat(std::numeric_limits<float>::infinity()), x);
        }
        v_elseif(normalized_exponent <= 0) { normalized = copysgn(vFloat(0.0f), x); }
        v_else { normalized = setexp(x, normalized_exponent); }
        v_endif;
        vFloat residual = normalized - 1.0f;
        vFloat log_mantissa = residual * selected_stirling_log_ratio(residual);
        vFloat exponent_ln2 = exponent_float * 0.6931471805599453f;
        vFloat log_x = log_mantissa + exponent_ln2;
        vFloat count_log = log_x * count;
        vFloat count_log_minus_count = count_log - count;
        vFloat scaled_main = x * count_log_minus_count;
        vFloat offset_log = log_x * log_offset;
        vFloat without_log_offset = scaled_main - offset_log;
        vFloat stirling_main = without_log_offset + constant;
        vFloat inverse_x = ckernel::sfpu::sfpu_reciprocal_iter<2>(x);
        vFloat correction_coordinate = inverse_x * correction_scale;
        vFloat correction = selected_stirling_correction(correction_coordinate);
        y = stirling_main + correction;
    }
    v_elseif(x >= overflow) { y = std::numeric_limits<float>::infinity(); }
    v_endif;

    constexpr float core_lower = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_LOWER_BITS);
    v_if(is_nan(x)) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x == std::numeric_limits<float>::infinity()) { y = std::numeric_limits<float>::infinity(); }
    v_elseif(x == -std::numeric_limits<float>::infinity()) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(is_zero(x)) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x <= core_lower) {
        y = std::numeric_limits<float>::quiet_NaN();
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
