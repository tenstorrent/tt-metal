// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Closed, ahead-of-time lowering for typed raw-coordinate domain actions.
// The generated adhoc header owns TT_DOMAIN_ACTION_DATA; this file owns only
// the structural SFPU application. Include it from inside namespace sfpi.

#pragma once

#include "sfpi_min_max.h"

#if defined(TT_SELECTED_CORE_AGGREGATE_STIRLING)
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

#if defined(TT_SELECTED_AGGREGATE_PUBLIC_LGAMMA_CLASS_REPAIR)
// Abstract interpretation of the public BF16 lgamma graph's result class.
// In the lower-domain region reached by this repair, a term is nonfinite iff
// its materialized operand is zero/subnormal or a non-positive integer.  The
// masks below test integrality from the operand's exponent/mantissa; no
// activation identity or fitted coefficient participates.
inline vInt selected_public_lgamma_lower_nonfinite(vFloat argument) {
    vFloat magnitude = setsgn(argument, 0);
    vInt exponent = exexp(magnitude, ExponentMode::Biased);
    vInt mantissa = exman(magnitude);
    vInt nonfinite = 0;
    v_if(exponent == 0) { nonfinite = 1; }
    v_elseif(argument < 0.0f) {
        v_if(exponent >= 134) { nonfinite = 1; }
        v_elseif(exponent == 133 && (mantissa & 0x0001ffff) == 0) { nonfinite = 1; }
        v_elseif(exponent == 132 && (mantissa & 0x0003ffff) == 0) { nonfinite = 1; }
        v_elseif(exponent == 131 && (mantissa & 0x0007ffff) == 0) { nonfinite = 1; }
        v_elseif(exponent == 130 && (mantissa & 0x000fffff) == 0) { nonfinite = 1; }
        v_elseif(exponent == 129 && (mantissa & 0x001fffff) == 0) { nonfinite = 1; }
        v_elseif(exponent == 128 && (mantissa & 0x003fffff) == 0) { nonfinite = 1; }
        v_elseif(exponent == 127 && mantissa == 0) { nonfinite = 1; }
        v_endif;
    }
    v_endif;
    return nonfinite;
}

inline vInt selected_public_shifted_lgamma_any_nonfinite(vFloat x) {
    vInt any_nonfinite = 0;
#pragma GCC unroll 4
    for (uint32_t term = 0; term < TT_SELECTED_AGGREGATE_CLASS_TERM_COUNT; ++term) {
        vFloat shift = __builtin_bit_cast(float, TT_SELECTED_AGGREGATE_CLASS_SHIFT_BITS[term]);
        vFloat argument = x - shift;
        argument = convert<vFloat16b>(argument, RoundMode::Nearest);
        any_nonfinite |= selected_public_lgamma_lower_nonfinite(argument);
    }
    // The target binary FPU and SFPU have distinct cancellation windows when
    // a small BF16 input is subtracted from the declared pole-producing
    // scalar. Replace only that window with the target-boundary relation.
    constexpr float scope_lower = __builtin_bit_cast(float, TT_SELECTED_AGGREGATE_BINARY_POLE_SFPU_SCOPE_BITS[0]);
    constexpr float scope_upper = __builtin_bit_cast(float, TT_SELECTED_AGGREGATE_BINARY_POLE_SFPU_SCOPE_BITS[1]);
    constexpr float target_lower = __builtin_bit_cast(float, TT_SELECTED_AGGREGATE_BINARY_POLE_TARGET_OPEN_BITS[0]);
    constexpr float target_upper = __builtin_bit_cast(float, TT_SELECTED_AGGREGATE_BINARY_POLE_TARGET_OPEN_BITS[1]);
    v_if(x > scope_lower && x < scope_upper) {
        any_nonfinite = 0;
        v_if(x > target_lower && x < target_upper) { any_nonfinite = 1; }
        v_endif;
    }
    v_endif;
    return any_nonfinite;
}

inline vInt selected_public_fold_zero_representative(vFloat x) {
    vInt is_zero_representative = 0;
#pragma GCC unroll 4
    for (uint32_t index = 0; index < TT_SELECTED_AGGREGATE_ZERO_REPRESENTATIVE_COUNT; ++index) {
        constexpr bool has_zero_representatives = TT_SELECTED_AGGREGATE_ZERO_REPRESENTATIVE_COUNT > 0;
        if constexpr (has_zero_representatives) {
            vFloat candidate = __builtin_bit_cast(float, TT_SELECTED_AGGREGATE_ZERO_REPRESENTATIVE_BITS[index]);
            v_if(x == candidate) { is_zero_representative = 1; }
            v_endif;
        }
    }
    return is_zero_representative;
}
#endif

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
#if defined(TT_SELECTED_AGGREGATE_PUBLIC_LGAMMA_CLASS_REPAIR)
        vInt any_nonfinite = selected_public_shifted_lgamma_any_nonfinite(x);
        vInt zero_representative = selected_public_fold_zero_representative(x);
        v_if(zero_representative != 0) { y = 0.0f; }
        v_elseif(any_nonfinite != 0) { y = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
#else
        y = std::numeric_limits<float>::quiet_NaN();
#endif
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) && !defined(TT_SELECTED_TTI_RAW_SHADOW_LATE_SCHEDULE) &&       \
    !defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION) && !defined(TT_SELECTED_COMPONENT_SFPI_LOOP_ROW) && \
    !defined(TT_ZONE_FACTORED_EXP_ROOT_LATE_RAW_SCHEDULE)
#error "raw BF16 NaN discrimination requires an admitted same-row structural schedule"
#endif

#if defined(TT_SELECTED_CORE_TOTAL_FORM)
template <uint32_t DEGREE>
inline vFloat selected_core_total_poly(const float* coefficients, vFloat coordinate) {
    vFloat result = coefficients[DEGREE];
#pragma GCC unroll 16
    for (int index = (int)DEGREE - 1; index >= 0; --index) {
        result = result * coordinate + coefficients[index];
    }
    return result;
}

inline vFloat selected_core_total_scale_pow2(vFloat value, vInt exponent) {
    vFloat result = value;
    vInt result_exponent = exexp(value, ExponentMode::Biased) + exponent;
    v_if(is_nan(value)) { result = value; }
    v_elseif(exexp(value, ExponentMode::Biased) == 255) { result = value; }
    v_elseif(is_zero(value)) { result = value; }
    v_elseif(result_exponent >= 255) { result = copysgn(vFloat(std::numeric_limits<float>::infinity()), value); }
#if defined(TT_SELECTED_CORE_SCALED_EXP_CORRECTION_TAIL_1)
    v_elseif(result_exponent == 0) {
        vFloat magnitude = setsgn(value, 0);
        vFloat normalized = setexp(magnitude, 127);
        v_if(normalized >= TT_SELECTED_CORE_UNDERFLOW_HALFWAY_MANTISSA) {
            result = copysgn(vFloat(std::numeric_limits<float>::min()), value);
        }
        v_else { result = copysgn(vFloat(0.0f), value); }
        v_endif;
    }
    v_elseif(result_exponent < 0) {
#else
    v_elseif(result_exponent <= 0) {
#endif
        result = copysgn(vFloat(0.0f), value);
    }
    v_else { result = setexp(value, result_exponent); }
    v_endif;
    return result;
}

#if defined(TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1) || defined(TT_SELECTED_CORE_SYMMETRIC_EXP_TAIL_1) || \
    defined(TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1) || defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1)
inline vFloat selected_core_total_exp(vFloat x) {
    vFloat xlog2 = x * TT_SELECTED_CORE_EXP2_MULT + (float)TT_SELECTED_CORE_EXP2_BIAS;
    vFloat lower = 0.0f;
    vFloat upper = 255.0f;
    ordered_min_max(lower, xlog2);
    ordered_min_max(xlog2, upper);
    vInt exponent = exexp(xlog2);
    vInt mantissa = exman(xlog2, MantissaMode::ImplicitOne);
    mantissa = shft(mantissa, exponent, ShiftMode::Logical);
    vFloat packed = as<vFloat>(mantissa);
    vInt integer_exponent = exexp(packed, ExponentMode::Biased);
    vMag fraction_mantissa = exman(packed);
    vFloat fraction = convert<vFloat>(fraction_mantissa, RoundMode::Nearest) * 0x1p-23f;
    vFloat exponential = selected_core_total_poly<TT_SELECTED_CORE_EXP_DEGREE>(TT_SELECTED_CORE_EXP_COEFFS, fraction);
#if defined(TT_SELECTED_CORE_EXP2_SHIFTED)
    return exponential * as<vFloat>(integer_exponent << 23);
#else
    vInt polynomial_exponent = exexp(exponential, ExponentMode::Biased);
    return setexp(exponential, integer_exponent + polynomial_exponent - 127 + TT_SELECTED_CORE_EXP2_OUTPUT_SHIFT);
#endif
}
#endif

#if defined(TT_SELECTED_CORE_ONE_SIDED_CODY_EXP_TAIL_1) || defined(TT_SELECTED_CORE_SIGMOID_PRODUCT_DERIVATIVE_TAIL_1)
inline vFloat selected_core_total_cody_exp(vFloat x) {
    vFloat argument = x * TT_SELECTED_CORE_EXP_ARGUMENT_SCALE;
    auto rounded = sfpi::round(argument * TT_SELECTED_CORE_INV_LN2);
    vFloat reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_HI + argument;
    reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_LO + reduced;
    vFloat exponential = selected_core_total_poly<TT_SELECTED_CORE_EXP_DEGREE>(TT_SELECTED_CORE_EXP_COEFFS, reduced);
    return selected_core_total_scale_pow2(exponential * TT_SELECTED_CORE_EXP_OUTPUT_SCALE, rounded.second);
}

#endif

#if defined(TT_SELECTED_CORE_BRIDGE_DIRECT_LOG_TAIL_1) || defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1)
inline vFloat selected_core_total_direct_log(vFloat x) {
    static_assert(TT_SELECTED_CORE_LOG_DEGREE == 3u);
    vInt exponent = as<vInt>(vFloat(0.75f));
    exponent = as<vInt>(x) - exponent;
    exponent = as<vInt>(setman(as<vFloat>(exponent), 0));
    vFloat residual = as<vFloat>(as<vInt>(x) - exponent) - 1.0f;
    vFloat result = TT_SELECTED_CORE_LOG_COEFFS[3];
    result = result * residual + TT_SELECTED_CORE_LOG_COEFFS[2];
    result = result * residual + TT_SELECTED_CORE_LOG_COEFFS[1];
    result = result * residual + TT_SELECTED_CORE_LOG_COEFFS[0];
    result = residual + (residual * residual) * result;
    vFloat exponent_float = convert<vFloat>(abs(exponent), RoundMode::Nearest);
    exponent_float = copysgn(exponent_float, as<vFloat>(exponent));
    result = exponent_float * 0x1.62e43p-24f + result;
#if !defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1)
    v_if(is_nan(x) || is_inf(x)) { result = setman(x, 0); }
    v_endif;
#endif
    return result;
}
#endif

#if defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
inline vFloat selected_core_total_exponent_replay(vFloat x, vFloat core) {
    vInt biased = exexp(x, ExponentMode::Biased);
    vFloat exponent = convert<vFloat>(as<vSMag>(biased - TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT), RoundMode::Nearest);
    vFloat result = exponent * TT_SELECTED_CORE_LOG_TWO + core;
    v_if(biased == 255) { result *= x; }
    v_endif;
    return result;
}

inline vFloat selected_core_exponent_replay_prepare(vFloat x, vInt& exponent_delta) {
    exponent_delta = 0;
    vFloat coordinate = x;
    v_if(coordinate > TT_SELECTED_CORE_UPPER) {
        exponent_delta = exexp(coordinate, ExponentMode::Biased) - TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT;
        coordinate = setexp(coordinate, TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT);
    }
    v_endif;
    vFloat lower = TT_SELECTED_CORE_LOWER;
    ordered_min_max(lower, coordinate);
    return coordinate;
}

inline void selected_core_exponent_replay_finalize(vFloat x, vInt exponent_delta, vFloat& result) {
    vFloat exponent = convert<vFloat>(as<vSMag>(exponent_delta), RoundMode::Nearest);
    result = exponent * TT_SELECTED_CORE_LOG_TWO + result;
    v_if(x < TT_SELECTED_CORE_INVALID_BELOW) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x < TT_SELECTED_CORE_BRIDGE_UPPER) {
        result = (x - TT_SELECTED_CORE_ENDPOINT) * TT_SELECTED_CORE_BRIDGE_SLOPE;
    }
    v_endif;
    constexpr int special_delta = 255 - TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT;
    v_if(exponent_delta == special_delta) { result *= x; }
    v_endif;
}
#endif

inline void apply_selected_core_total_form(vFloat x, vFloat& result) {
#if defined(TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1)
#if defined(TT_SELECTED_CORE_TTI_SQUARE_AFFINE_RECONSTRUCTION)
    // The replay body has already formed scale*x*q(x^2)+bias.  Its same-row
    // suffix owns only the typed exterior zones and raw-class repair.
    v_if(x <= TT_SELECTED_CORE_NEGATIVE_TERMINAL) {
#else
    v_if(x >= TT_SELECTED_CORE_LOWER && x < TT_SELECTED_CORE_UPPER) {
        // The selected leaf is q(x^2); the typed component schedule owns the
        // affine reconstruction f(x)=scale*x*q(x^2)+bias independently of
        // which evaluator executes the leaf.
        result = (__builtin_bit_cast(float, TT_SELECTED_CORE_OUTPUT_SCALE_BITS) * x) * result +
                 __builtin_bit_cast(float, TT_SELECTED_CORE_OUTPUT_BIAS_BITS);
    }
    v_elseif(x <= TT_SELECTED_CORE_NEGATIVE_TERMINAL) {
#endif
        result = setsgn(vFloat(0.0f), 1);
    }
    v_elseif(x < TT_SELECTED_CORE_LOWER) {
        vFloat square = x * x;
        vFloat exponential = selected_core_total_exp(square * TT_SELECTED_CORE_EXPONENT_SCALE);
        vFloat reciprocal = ckernel::sfpu::sfpu_reciprocal_iter<TT_SELECTED_CORE_RECIPROCAL_ITERS>(square);
        vFloat lower = TT_SELECTED_CORE_CORRECTION_MIN;
        vFloat upper = TT_SELECTED_CORE_CORRECTION_MAX;
        ordered_min_max(lower, reciprocal);
        ordered_min_max(reciprocal, upper);
        vFloat correction = selected_core_total_poly<TT_SELECTED_CORE_CORRECTION_DEGREE>(
            TT_SELECTED_CORE_CORRECTION_COEFFS, reciprocal);
        result = exponential * ((x * correction) * TT_SELECTED_CORE_CORRECTION_SCALE);
    }
    v_elseif(x >= TT_SELECTED_CORE_POSITIVE_IDENTITY) { result = 1.0f; }
    v_endif;
#elif defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1)
    vFloat magnitude = setsgn(x, 0);
    v_if(magnitude >= TT_SELECTED_CORE_TERMINAL_MAGNITUDE) { result = 0.0f; }
    v_elseif(magnitude >= TT_SELECTED_CORE_UPPER) {
        vFloat argument = magnitude * TT_SELECTED_CORE_EXP_ARGUMENT_SCALE + TT_SELECTED_CORE_EXP_ARGUMENT_ADDEND;
        result = selected_core_total_exp(argument);
    }
    v_endif;
#elif defined(TT_SELECTED_CORE_SIGMOID_PRODUCT_DERIVATIVE_TAIL_1)
    vFloat magnitude = setsgn(x, 0);
    v_if(x <= TT_SELECTED_CORE_NEGATIVE_TERMINAL) { result = copysgn(vFloat(0.0f), x); }
    v_elseif(magnitude > TT_SELECTED_CORE_UPPER && x < TT_SELECTED_CORE_POSITIVE_IDENTITY) {
        // This reduction deliberately stays in the caller.  The SFPU register
        // allocator treats a separately inlined reduction helper differently
        // across the following rational graph; keeping the structural nodes
        // in one scope reproduces the verified target-lattice schedule.
        vFloat negative_one = -1.0f;
        vFloat argument = magnitude * negative_one;
        vFloat scaled = argument * TT_SELECTED_CORE_INV_LN2;
        auto rounded = sfpi::round(scaled);
        vFloat reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_HI + argument;
        reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_LO + reduced;
        vFloat exponential =
            selected_core_total_poly<TT_SELECTED_CORE_EXP_DEGREE>(TT_SELECTED_CORE_EXP_COEFFS, reduced);
        vFloat scaled_exponential_unscaled = exponential * TT_SELECTED_CORE_EXP_OUTPUT_SCALE;
        vFloat scaled_exponential = scaled_exponential_unscaled;
        vInt scaled_exponent = exexp(scaled_exponential_unscaled, ExponentMode::Biased) + rounded.second;
        // This typed tail is reached only for finite |x| between the selected
        // core and negative terminal.  Its exp polynomial is positive and
        // finite, and range reduction cannot overflow while reconstructing
        // exp(-|x|); only the underflow boundary needs a branch.  Keeping the
        // impossible generic NaN/Inf/overflow arms here made latest SFPI GCC's
        // rvtt_vif pass crash after inlining the surrounding rational graph.
        v_if(scaled_exponent <= 0) { scaled_exponential = copysgn(vFloat(0.0f), scaled_exponential_unscaled); }
        v_else { scaled_exponential = setexp(scaled_exponential_unscaled, scaled_exponent); }
        v_endif;
        vFloat denominator = TT_SELECTED_CORE_COMMON_SCALE + scaled_exponential;
        vFloat inverse_denominator =
            ckernel::sfpu::sfpu_reciprocal_iter<TT_SELECTED_CORE_RECIPROCAL_ITERS>(denominator);
        vFloat quotient = scaled_exponential * inverse_denominator;
        vFloat one_minus_quotient = 1.0f - quotient;
        vFloat symmetric_derivative = quotient * one_minus_quotient;
        vFloat input_times_derivative = x * symmetric_derivative;
        vFloat positive_factor = one_minus_quotient + input_times_derivative;
        vFloat inverse_denominator_square = inverse_denominator * inverse_denominator;
        vFloat scaled_input = x * TT_SELECTED_CORE_COMMON_SCALE;
        vFloat shifted_denominator = denominator + scaled_input;
        vFloat negative_rational_scale = shifted_denominator * inverse_denominator_square;
        vFloat negative_factor = scaled_exponential * negative_rational_scale;
        vFloat nonnegative_selector = x + magnitude;
        v_if(is_zero(nonnegative_selector)) { result = negative_factor; }
        v_else { result = positive_factor; }
        v_endif;
    }
    v_elseif(x >= TT_SELECTED_CORE_POSITIVE_IDENTITY) { result = 1.0f; }
    v_endif;
#elif defined(TT_SELECTED_CORE_SYMMETRIC_EXP_TAIL_1)
    v_if(x < TT_SELECTED_CORE_LOWER || x > TT_SELECTED_CORE_UPPER) {
        vFloat argument = setsgn(x, 1);
        vFloat lower = TT_SELECTED_CORE_NEGATIVE_TERMINAL;
        ordered_min_max(lower, argument);
        result = selected_core_total_exp(argument);
    }
    v_endif;
#elif defined(TT_SELECTED_CORE_ONE_SIDED_CODY_EXP_TAIL_1)
#if TT_SELECTED_CORE_EXP_TAIL_ABOVE
    v_if(x < TT_SELECTED_CORE_LOWER) { result = TT_SELECTED_CORE_OPPOSITE_VALUE; }
    v_elseif(x > TT_SELECTED_CORE_UPPER && x < TT_SELECTED_CORE_TERMINAL) { result = selected_core_total_cody_exp(x); }
    v_elseif(x >= TT_SELECTED_CORE_TERMINAL) { result = TT_SELECTED_CORE_TERMINAL_VALUE; }
    v_endif;
#else
    v_if(x <= TT_SELECTED_CORE_TERMINAL) { result = TT_SELECTED_CORE_TERMINAL_VALUE; }
    v_elseif(x < TT_SELECTED_CORE_LOWER) { result = selected_core_total_cody_exp(x); }
    v_elseif(x > TT_SELECTED_CORE_UPPER) { result = TT_SELECTED_CORE_OPPOSITE_VALUE; }
    v_endif;
#endif
#elif defined(TT_SELECTED_CORE_ONE_SIDED_EXP_TAIL_1)
    v_if(x <= TT_SELECTED_CORE_NEGATIVE_TERMINAL) { result = 0.0f; }
    v_elseif(x < TT_SELECTED_CORE_LOWER) { result = selected_core_total_exp(x); }
    v_elseif(x > TT_SELECTED_CORE_UPPER) { result = TT_SELECTED_CORE_POSITIVE_VALUE; }
    v_endif;
#elif defined(TT_SELECTED_CORE_BRIDGE_DIRECT_LOG_TAIL_1)
    v_if(x < TT_SELECTED_CORE_INVALID_BELOW) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x == TT_SELECTED_CORE_ENDPOINT) { result = TT_SELECTED_CORE_ENDPOINT_VALUE; }
    v_elseif(x < TT_SELECTED_CORE_BRIDGE_UPPER) { result = TT_SELECTED_CORE_BRIDGE_VALUE; }
    v_elseif(x > TT_SELECTED_CORE_UPPER) { result = selected_core_total_direct_log(x) + TT_SELECTED_CORE_TAIL_ADDEND; }
    v_endif;
#elif defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
    v_if(x < TT_SELECTED_CORE_INVALID_BELOW) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x < TT_SELECTED_CORE_BRIDGE_UPPER) {
        result = (x - TT_SELECTED_CORE_ENDPOINT) * TT_SELECTED_CORE_BRIDGE_SLOPE;
    }
    v_elseif(x > TT_SELECTED_CORE_UPPER) { result = selected_core_total_exponent_replay(x, result); }
    v_endif;
#elif defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1)
    v_if(is_inf(x)) { result = x; }
    v_elseif(x < TT_SELECTED_CORE_LOWER || x > TT_SELECTED_CORE_UPPER) {
        vFloat magnitude = setsgn(x, 0);
        magnitude = selected_core_total_direct_log(magnitude) + TT_SELECTED_CORE_TAIL_ADDEND;
        result = copysgn(magnitude, x);
    }
    v_endif;
#elif defined(TT_SELECTED_CORE_GAUSSIAN_MILLS_TAIL_1)
    v_if(x < TT_SELECTED_CORE_LOWER) {
        vFloat exponent_coordinate = (x * x) * TT_SELECTED_CORE_EXPONENT_SCALE;
        auto rounded = sfpi::round(exponent_coordinate * TT_SELECTED_CORE_INV_LN2);
        vFloat reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_HI + exponent_coordinate;
        reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_LO + reduced;
        vFloat exponential =
            selected_core_total_poly<TT_SELECTED_CORE_EXP_DEGREE>(TT_SELECTED_CORE_EXP_COEFFS, reduced);
        vFloat reciprocal = ckernel::sfpu::sfpu_reciprocal_iter<TT_SELECTED_CORE_RECIPROCAL_ITERS>(setsgn(x, 0));
        vFloat correction = selected_core_total_poly<TT_SELECTED_CORE_CORRECTION_DEGREE>(
            TT_SELECTED_CORE_CORRECTION_COEFFS, reciprocal * reciprocal);
        vFloat magnitude = selected_core_total_scale_pow2(
            exponential * (correction * TT_SELECTED_CORE_CORRECTION_SCALE), rounded.second);
        result = -magnitude;
    }
    v_endif;
    v_if(x <= TT_SELECTED_CORE_NEGATIVE_TERMINAL) { result = setsgn(vFloat(0.0f), 1); }
    v_elseif(x > TT_SELECTED_CORE_UPPER && x < TT_SELECTED_CORE_POSITIVE_IDENTITY) {
        vFloat exponent_coordinate = (x * x) * TT_SELECTED_CORE_EXPONENT_SCALE;
        auto rounded = sfpi::round(exponent_coordinate * TT_SELECTED_CORE_INV_LN2);
        vFloat reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_HI + exponent_coordinate;
        reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_LO + reduced;
        vFloat exponential =
            selected_core_total_poly<TT_SELECTED_CORE_EXP_DEGREE>(TT_SELECTED_CORE_EXP_COEFFS, reduced);
        vFloat reciprocal = ckernel::sfpu::sfpu_reciprocal_iter<TT_SELECTED_CORE_RECIPROCAL_ITERS>(x);
        vFloat correction = selected_core_total_poly<TT_SELECTED_CORE_CORRECTION_DEGREE>(
            TT_SELECTED_CORE_CORRECTION_COEFFS, reciprocal * reciprocal);
        vFloat magnitude = selected_core_total_scale_pow2(
            exponential * (correction * TT_SELECTED_CORE_CORRECTION_SCALE), rounded.second);
        result = x - magnitude;
    }
    v_elseif(x >= TT_SELECTED_CORE_POSITIVE_IDENTITY) { result = x; }
    v_endif;
#elif defined(TT_SELECTED_CORE_SCALED_EXP_CORRECTION_TAIL_1)
    vFloat magnitude = setsgn(x, 0);
    v_if(magnitude >= TT_SELECTED_CORE_TERMINAL_MAGNITUDE) { result = std::numeric_limits<float>::infinity(); }
    v_elseif(magnitude > TT_SELECTED_CORE_UPPER) {
        auto rounded = sfpi::round(magnitude * TT_SELECTED_CORE_INV_LN2);
        vFloat reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_HI + magnitude;
        reduced = rounded.first * TT_SELECTED_CORE_NEG_LN2_LO + reduced;
        vFloat exponential =
            selected_core_total_poly<TT_SELECTED_CORE_EXP_DEGREE>(TT_SELECTED_CORE_EXP_COEFFS, reduced);
        vFloat reciprocal = ckernel::sfpu::sfpu_reciprocal_iter<TT_SELECTED_CORE_RECIPROCAL_ITERS>(magnitude);
        vFloat correction = selected_core_total_poly<TT_SELECTED_CORE_CORRECTION_DEGREE>(
            TT_SELECTED_CORE_CORRECTION_COEFFS, reciprocal);
        result = selected_core_total_scale_pow2(exponential * correction, rounded.second);
    }
    v_endif;
#else
#error "unknown selected-core total form"
#endif
}

#endif

inline vFloat prepare_raw_domain_input(vFloat x_raw) {
#if defined(TT_SELECTED_CORE_AGGREGATE_STIRLING)
    constexpr float core_lower = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_LOWER_BITS);
    constexpr float core_upper = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_UPPER_BITS);
    vFloat prepared = x_raw;
    v_if(prepared < core_lower) { prepared = core_lower; }
    v_elseif(prepared > core_upper) { prepared = core_upper; }
    v_endif;
    return prepared;
#elif defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
    vFloat coordinate = x_raw;
    v_if(coordinate > TT_SELECTED_CORE_UPPER) {
        coordinate = setexp(coordinate, TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT);
    }
    v_endif;
    vFloat lower = TT_SELECTED_CORE_LOWER;
    ordered_min_max(lower, coordinate);
    return coordinate;
#elif defined(TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1)
    vFloat coordinate = x_raw * x_raw;
    vFloat lower = 0.0f;
    vFloat upper = TT_SELECTED_CORE_UPPER * TT_SELECTED_CORE_UPPER;
    ordered_min_max(lower, coordinate);
    ordered_min_max(coordinate, upper);
    return coordinate;
#elif defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1)
    vFloat coordinate = (x_raw * x_raw) * TT_SELECTED_CORE_COORDINATE_SCALE + TT_SELECTED_CORE_COORDINATE_BIAS;
    vFloat lower = -1.0f;
    vFloat upper = 1.0f;
    ordered_min_max(lower, coordinate);
    ordered_min_max(coordinate, upper);
    return coordinate;
#elif defined(TT_SELECTED_CORE_TOTAL_FORM)
    vFloat coordinate = x_raw;
    v_if(coordinate < TT_SELECTED_CORE_LOWER) { coordinate = TT_SELECTED_CORE_LOWER; }
    v_elseif(coordinate > TT_SELECTED_CORE_UPPER) { coordinate = TT_SELECTED_CORE_UPPER; }
    v_endif;
    return coordinate;
#elif defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP)
    // A one-sided constant terminal can be folded into an ingress maximum and
    // an egress maximum.  Only the opposite raw action remains below.
    return sfpi::max(x_raw, TT_DOMAIN_ACTION_LOWER_CLAMP_BOUND);
#elif defined(TT_DOMAIN_ACTION_TERMINAL_INGRESS_ONLY)
    // Both typed terminals are already the selected core's values at its
    // callable endpoints, as proved over the complete physical BF16 lattice.
    vFloat prepared = sfpi::max(x_raw, TT_DOMAIN_ACTION_CALLABLE_LOWER_BOUND);
    return sfpi::min(prepared, TT_DOMAIN_ACTION_CALLABLE_UPPER_BOUND);
#elif defined(TT_DOMAIN_ACTION_RAW_TERMINAL_ENVELOPE)
    // The compiler emits this only for one lower/upper pair whose untouched
    // interval is contained in the selected CSV's callable domain.  Evaluation
    // is therefore always in-domain; finalization below replays the terminal
    // actions from the original raw coordinate.
    vFloat prepared = sfpi::max(x_raw, TT_DOMAIN_ACTION_RAW_LOWER_BOUND);
    return sfpi::min(prepared, TT_DOMAIN_ACTION_RAW_UPPER_BOUND);
#else
    return x_raw;
#endif
}

#if defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RAW_NEG_EXP_FF_INGRESS)
#if !defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP)
#error "negative-NaN ingress consumption requires the proved lower clamp"
#endif
inline vFloat prepare_raw_domain_input(vFloat x_raw, vUInt raw_u16) {
    // Restore the encoded sign before the already-proved ingress maximum.
    // Blackhole's BF16 transport maps either NaN sign to +Inf, while every
    // finite negative value already carries this sign.  Applying the encoded
    // sign to the complete negative half is therefore idempotent for finite
    // inputs and turns negative NaN/Inf into the same lower-clamp input.  The
    // following shape-derived maximum still owns the terminal value.
    // The U16 DST view is byte-swapped relative to the IEEE BF16 word, so
    // its encoded sign occupies bit 7 (the same layout used by the generic
    // raw-terminal discriminators below).
    vUInt raw_sign = raw_u16 & vUInt(0x0080u);
    v_if(raw_sign == vUInt(0x0080u)) { x_raw = setsgn(x_raw, 1); }
    v_endif;
    return prepare_raw_domain_input(x_raw);
}
#endif

#if defined(TT_DOMAIN_ACTION_PROGRAM)

#if defined(TT_DOMAIN_ACTION_MIRRORED_CLASS_TERMINALS)

// Two structurally matched mirrored exterior NaN records collapse to one
// magnitude predicate. When present, the inclusive endpoint records share
// that magnitude and restore only the input sign on infinity.
inline void apply_raw_pre_domain_actions(vFloat x_raw, vFloat& result) {
    constexpr auto lower_exterior = TT_DOMAIN_ACTION_DATA[0];
    constexpr auto upper_exterior = TT_DOMAIN_ACTION_DATA[1];
    static_assert(
        lower_exterior.direction == 0 && lower_exterior.inclusive == 0 && lower_exterior.action_kind == 3 &&
            lower_exterior.return_class == 0 && lower_exterior.bound == -TT_DOMAIN_ACTION_MIRRORED_BOUND,
        "mirrored lower exterior action drifted");
    static_assert(
        upper_exterior.direction == 1 && upper_exterior.inclusive == 0 && upper_exterior.action_kind == 3 &&
            upper_exterior.return_class == 0 && upper_exterior.bound == TT_DOMAIN_ACTION_MIRRORED_BOUND,
        "mirrored upper exterior action drifted");
#if defined(TT_DOMAIN_ACTION_MIRRORED_SIGNED_ENDPOINTS)
    static_assert(TT_DOMAIN_ACTION_COUNT == 4, "signed endpoints require four actions");
    constexpr auto lower_endpoint = TT_DOMAIN_ACTION_DATA[2];
    constexpr auto upper_endpoint = TT_DOMAIN_ACTION_DATA[3];
    static_assert(
        lower_endpoint.direction == 0 && lower_endpoint.inclusive != 0 && lower_endpoint.action_kind == 3 &&
            lower_endpoint.return_class == 2 && lower_endpoint.bound == -TT_DOMAIN_ACTION_MIRRORED_BOUND,
        "mirrored lower endpoint action drifted");
    static_assert(
        upper_endpoint.direction == 1 && upper_endpoint.inclusive != 0 && upper_endpoint.action_kind == 3 &&
            upper_endpoint.return_class == 1 && upper_endpoint.bound == TT_DOMAIN_ACTION_MIRRORED_BOUND,
        "mirrored upper endpoint action drifted");
#else
    static_assert(TT_DOMAIN_ACTION_COUNT == 2, "exterior-only schedule requires two actions");
#endif

    vFloat absolute_raw = setsgn(x_raw, 0);
    v_if(absolute_raw > TT_DOMAIN_ACTION_MIRRORED_BOUND) { result = std::numeric_limits<float>::quiet_NaN(); }
#if defined(TT_DOMAIN_ACTION_MIRRORED_SIGNED_ENDPOINTS)
    v_elseif(absolute_raw >= TT_DOMAIN_ACTION_MIRRORED_BOUND) {
        vFloat endpoint_inf = std::numeric_limits<float>::infinity();
        result = copysgn(endpoint_inf, x_raw);
    }
#endif
    v_endif;
}

#elif defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP)

// The lower constant action is owned by the ingress/egress max pair.  Retain
// exactly the opposite typed raw action, with its compile-time action shape.
inline void apply_raw_pre_domain_actions(vFloat x_raw, vFloat& result) {
    static_assert(TT_DOMAIN_ACTION_COUNT == 2, "one-sided lowering requires two actions");
    constexpr auto lower = TT_DOMAIN_ACTION_DATA[TT_DOMAIN_ACTION_LOWER_RECORD_INDEX];
    constexpr auto upper = TT_DOMAIN_ACTION_DATA[TT_DOMAIN_ACTION_UPPER_RECORD_INDEX];
    static_assert(
        lower.direction == 0 && lower.action_kind == 0, "lower ingress/egress fold requires a constant below action");
    static_assert(
        lower.bound == TT_DOMAIN_ACTION_LOWER_CLAMP_BOUND && lower.value == TT_DOMAIN_ACTION_LOWER_TERMINAL_VALUE,
        "lower fold certificate does not match typed action");
    static_assert(
        upper.direction == 1 && upper.action_kind <= 2, "retained upper action must be constant, identity, or affine");

#if defined(TT_DOMAIN_ACTION_UPPER_FINITE_INTRINSIC)
    // S55 proved the selected exterior leaf realizes this action on every
    // finite physical BF16 encoding. The typed special policy below retains
    // authority over exponent-FF inputs, so no upper branch remains here.
    (void)x_raw;
    (void)result;
#else
    vFloat action_value = upper.value;
    if constexpr (upper.action_kind == 1) {
        action_value = x_raw;
    } else if constexpr (upper.action_kind == 2) {
        vFloat action_scale = upper.scale;
        vFloat action_bias = upper.bias;
        action_value =
            __builtin_rvtt_sfpmad(x_raw.get(), action_scale.get(), action_bias.get(), SFPMAD_MOD1_OFFSET_NONE);
    }
    if constexpr (upper.inclusive != 0) {
        v_if(x_raw >= upper.bound) { result = action_value; }
        v_endif;
    } else {
        v_if(x_raw > upper.bound) { result = action_value; }
        v_endif;
    }
#endif
}

#elif defined(TT_DOMAIN_ACTION_SYMMETRIC_CONSTANT_RAW_TAILS)

// This macro is emitted only for the complete typed action shape certified by
// detect_symmetric_constant_raw_tails. Keep the original records and hashes in
// generated source: these assertions make the cheaper lowering subordinate to
// that unchanged program rather than a replacement contract.
inline void apply_raw_pre_domain_actions(vFloat x_raw, vFloat& result) {
    static_assert(TT_DOMAIN_ACTION_COUNT == 2, "symmetric tails require exactly two actions");
    constexpr auto first = TT_DOMAIN_ACTION_DATA[0];
    constexpr auto second = TT_DOMAIN_ACTION_DATA[1];
    static_assert(first.action_kind == 0 && second.action_kind == 0, "symmetric tails require constant actions");
    static_assert(first.inclusive == second.inclusive, "symmetric tails require matching inclusivity");
    static_assert(
        first.inclusive == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE,
        "symmetric tail certificate does not match action records");
    static_assert(
        (first.direction == 0 && second.direction == 1 && first.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
         second.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) ||
            (first.direction == 1 && second.direction == 0 && first.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
             second.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND),
        "symmetric tail certificate does not match action bounds");
    static_assert(
        first.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE && second.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE,
        "symmetric tail certificate does not match action values");

    vFloat absolute_raw = setsgn(x_raw, 0);
    if constexpr (TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE != 0) {
        v_if(absolute_raw >= TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) { result = TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE; }
        v_endif;
    } else {
        v_if(absolute_raw > TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) { result = TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE; }
        v_endif;
    }
}

#elif defined(TT_DOMAIN_ACTION_SYMMETRIC_SIGNED_CONSTANT_RAW_TAILS)

// Two exact opposite constants over symmetric raw-coordinate tails collapse
// to one magnitude predicate plus sign restoration.  The complete typed
// records remain embedded and statically checked; this is a lowering of that
// program, not a new function-specific terminal path.
inline void apply_raw_pre_domain_actions(vFloat x_raw, vFloat& result) {
    static_assert(TT_DOMAIN_ACTION_COUNT == 2, "signed tails require exactly two actions");
    constexpr auto first = TT_DOMAIN_ACTION_DATA[0];
    constexpr auto second = TT_DOMAIN_ACTION_DATA[1];
    static_assert(first.action_kind == 0 && second.action_kind == 0, "signed tails require constant actions");
    static_assert(first.inclusive == second.inclusive, "signed tails require matching inclusivity");
    static_assert(
        first.inclusive == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE,
        "signed-tail certificate does not match action records");
    static_assert(
        (first.direction == 0 && second.direction == 1 && first.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
         second.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
         first.value == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_MAGNITUDE &&
         second.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_MAGNITUDE) ||
            (first.direction == 1 && second.direction == 0 && first.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
             second.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
             first.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_MAGNITUDE &&
             second.value == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_MAGNITUDE),
        "signed-tail certificate does not match bounds or values");

    vFloat absolute_raw = setsgn(x_raw, 0);
    vFloat signed_value = copysgn(vFloat(TT_DOMAIN_ACTION_SYMMETRIC_TAIL_MAGNITUDE), x_raw);
#if defined(TT_DOMAIN_ACTION_FINITE_ONLY)
    static_assert(
        TT_DOMAIN_ACTION_TERMINAL_SUFFIX_SEMANTIC_ISSUES == 7u, "finite signed-tail suffix issue certificate drifted");
    // The suffix count above only sees the terminal constant, so it stays 7u
    // for any certified form whose terminal magnitude is the free LCONST_1
    // operand -- including forms whose denominator or numerator coefficients
    // are NOT unit and therefore need in-body immediates the body below is not
    // built for.  The total is the field that moves for every such form, so
    // assert it too: this is what refuses a certified-but-unlowerable
    // coefficient form at build time instead of shipping a body whose issue
    // accounting silently disagrees with its certificate.
    static_assert(
        TT_DOMAIN_ACTION_COMMON_SFPI_TOTAL_SEMANTIC_ISSUES == 19u,
        "finite signed-tail total issue certificate drifted");
    // The mathematical special policy runs after typed finite actions.  Keep
    // exponent-255 inputs under the selected evaluator's class authority:
    // ordered comparisons reject NaNs and this exact upper bound rejects
    // either infinity.  The certificate is required for this refinement.
    constexpr float max_finite = std::numeric_limits<float>::max();
    v_if(absolute_raw <= max_finite) {
#endif
        if constexpr (TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE != 0) {
            v_if(absolute_raw >= TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) { result = signed_value; }
            v_endif;
        } else {
            v_if(absolute_raw > TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) { result = signed_value; }
            v_endif;
        }
#if defined(TT_DOMAIN_ACTION_FINITE_ONLY)
    }
    v_endif;
#endif
}

#else

// Generated records use 0=below and 1=above. Recursing from the last record
// toward zero preserves the ABI's "first matching action wins" ordering: an
// earlier predicate overwrites a later match. Action dispatch is compile-time:
// no activation identity or per-operation result participates.
template <uint32_t INDEX>
inline void apply_raw_pre_domain_actions(vFloat x_raw, vFloat& result) {
    constexpr auto record = TT_DOMAIN_ACTION_DATA[INDEX];
    static_assert(record.action_kind <= 4, "unsupported domain-action kind");
    static_assert(record.return_class <= 4, "unsupported domain-action return class");
    vFloat action_value = record.value;
    if constexpr (record.action_kind == 1) {
        action_value = x_raw;
    } else if constexpr (record.action_kind == 2) {
        vFloat action_scale = record.scale;
        vFloat action_bias = record.bias;
        action_value =
            __builtin_rvtt_sfpmad(x_raw.get(), action_scale.get(), action_bias.get(), SFPMAD_MOD1_OFFSET_NONE);
    } else if constexpr (record.action_kind == 3) {
        constexpr float class_value = record.return_class == 0   ? std::numeric_limits<float>::quiet_NaN()
                                      : record.return_class == 1 ? std::numeric_limits<float>::infinity()
                                      : record.return_class == 2 ? -std::numeric_limits<float>::infinity()
                                      : record.return_class == 3 ? 0.0f
                                                                 : -0.0f;
        action_value = class_value;
    } else if constexpr (record.action_kind == 4) {
        vFloat magnitude = std::numeric_limits<float>::infinity();
        action_value = copysgn(magnitude, x_raw);
    }
    if constexpr (record.direction == 0 && record.inclusive != 0) {
        v_if(x_raw <= record.bound) { result = action_value; }
        v_endif;
    } else if constexpr (record.direction == 0) {
        v_if(x_raw < record.bound) { result = action_value; }
        v_endif;
    } else if constexpr (record.inclusive != 0) {
        v_if(x_raw >= record.bound) { result = action_value; }
        v_endif;
    } else {
        v_if(x_raw > record.bound) { result = action_value; }
        v_endif;
    }
    if constexpr (INDEX > 0) {
        apply_raw_pre_domain_actions<INDEX - 1>(x_raw, result);
    }
}

inline void apply_raw_pre_domain_actions(vFloat x_raw, vFloat& result) {
    static_assert(TT_DOMAIN_ACTION_COUNT > 0, "domain-action program must not be empty");
    apply_raw_pre_domain_actions<TT_DOMAIN_ACTION_COUNT - 1>(x_raw, result);
}

#endif  // compact symmetric raw-tail lowerings

#endif  // TT_DOMAIN_ACTION_PROGRAM

inline vFloat target_domain_action_coordinate(vFloat x_raw) {
#if defined(TT_TARGET_BH_BF16_ACTION_COORDINATE)
    // Float16_b exponent-zero inputs expand into ordinary FP32 values in DST.
    // The bound target declares DAZ before mathematical dispatch, so normalize
    // exactly that open interval here, once, before actions and finalization.
    // The two ordered comparisons also canonicalize either signed zero to +0.
    constexpr float min_normal = std::numeric_limits<float>::min();
    vFloat effective = x_raw;
    v_if(effective > -min_normal) { effective = 0.0f; }
    v_endif;
    v_if(x_raw >= min_normal) { effective = x_raw; }
    v_endif;
    return effective;
#else
    return x_raw;
#endif
}

#if defined(TT_TARGET_BH_BF16_PACK_RELU_RAW_NEG_NAN_REPAIR)
inline void apply_pack_relu_raw_neg_nan_repair(vUInt raw_u16, vFloat& result) {
    // STACC_RELU clears every sign-bit datum, including negative NaN payloads.
    // BH BF16 ingress maps both raw NaN signs to +Inf; restore that quotient
    // before packing so mode 1 passes +Inf and mode 3 applies its finite cap.
    // In the U16 DST view the bytes are swapped relative to an IEEE BF16
    // word: exponent[7] is bit 15, exponent[6:0] are bits 6:0, and sign is
    // bit 7.  0x80ff therefore denotes negative infinity; a nonzero
    // mantissa in bits 14:8 makes exactly the 127 negative NaN payloads.
    vUInt negative_special_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(negative_special_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = std::numeric_limits<float>::infinity(); }
        v_endif;
    }
    v_endif;
}
#endif

// Encoded-raw terminal values are independent of the mathematical special
// finalizer.  Some TTNN quotient plans need only a signed raw NaN/Inf/zero
// repair after the ordinary evaluator and therefore intentionally do not
// define TT_TARGET_BH_BF16_SPECIAL_FINALIZER.
#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NONFINITE_RESULT) ||     \
    defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT) || \
    defined(TT_TARGET_BH_BF16_RAW_POS_INF_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT) ||       \
    defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT) ||     \
    defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_RESULT)
template <int CODE>
inline vFloat target_raw_terminal_value(vFloat computed, float constant = 0.0f) {
    static_assert(CODE >= 0 && CODE <= 6, "unsupported raw terminal class");
    if constexpr (CODE == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    } else if constexpr (CODE == 1) {
        return std::numeric_limits<float>::infinity();
    } else if constexpr (CODE == 2) {
        return -std::numeric_limits<float>::infinity();
    } else if constexpr (CODE == 3) {
        return vFloat(0.0f);
    } else if constexpr (CODE == 4) {
        return setsgn(vFloat(0.0f), 1);
    } else if constexpr (CODE == 5) {
        return computed;
    } else {
        return vFloat(constant);
    }
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR)
inline void apply_target_raw_nonfinite_policy(vUInt raw_u16, vFloat& result) {
    // In the physical U16 DST view, exponent[7:0] is bits 7:0. Ignoring the
    // sign therefore selects exactly both infinities and all 254 NaN payloads.
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NONFINITE_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_NONFINITE_CONSTANT);
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR)
inline void apply_target_raw_pos_nonfinite_policy(vUInt raw_u16, vFloat& result) {
    // Physical BF16 layout has the complete exponent byte in bits 7:0 and
    // the sign in bit 15. This one quotient is exactly +Inf plus all 127
    // positive NaNs; codegen admits it only for equal typed constant actions.
    vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x80ffu);
    v_if(exponent_and_sign_delta == 0u) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_POS_NONFINITE_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_POS_NONFINITE_CONSTANT);
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT)
inline void apply_target_raw_signed_nonfinite_split(vUInt raw_u16, vFloat& result) {
    // Share the physical exponent-FF test between the positive constant
    // quotient and signed-ingress negative-NaN policy. Negative infinity is
    // left to the numeric body; a nonzero negative mantissa is the NaN arm.
    vUInt exponent = raw_u16 & vUInt(0x00ffu);
    v_if(exponent == vUInt(0x00ffu)) {
        vUInt sign = raw_u16 & vUInt(0x8000u);
        v_if(sign == 0u) {
            result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_POS_NONFINITE_RESULT>(
                result, TT_TARGET_BH_BF16_RAW_POS_NONFINITE_CONSTANT);
        }
        v_else {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR)
inline void apply_target_raw_pos_inf_policy(vUInt raw_u16, vFloat& result) {
    v_if(raw_u16 == vUInt(0x00ffu)) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_POS_INF_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_POS_INF_CONSTANT);
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR)
inline void apply_target_raw_neg_inf_policy(vUInt raw_u16, vFloat& result) {
    v_if(raw_u16 == vUInt(0x80ffu)) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_NEG_INF_CONSTANT);
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) && defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT)
inline void apply_target_raw_pos_nan_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x80ffu);
    v_if(exponent_and_sign_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) {
            result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT>(
                result, TT_TARGET_BH_BF16_RAW_POS_NAN_CONSTANT);
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) && defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT)
inline void apply_target_raw_neg_nan_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(exponent_and_sign_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) {
            result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT>(
                result, TT_TARGET_BH_BF16_RAW_NEG_NAN_CONSTANT);
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR)
inline void apply_target_raw_pos_zero_policy(vUInt raw_u16, vFloat& result) {
    v_if(raw_u16 == vUInt(0x0000u)) {
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_POS_ZERO_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_POS_ZERO_CONSTANT);
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
inline void apply_target_raw_subnormal_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent = raw_u16 & vUInt(0x00ffu);
    vUInt mantissa = raw_u16 & vUInt(0x7f00u);
    v_if((exponent == 0u) && (mantissa != 0u)) {
        vUInt sign = raw_u16 & vUInt(0x8000u);
#if defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR)
        v_if(sign == 0u) {
            result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_RESULT>(
                result, TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_CONSTANT);
        }
        v_endif;
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
        v_if(sign != 0u) {
            result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_RESULT>(
                result, TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_CONSTANT);
        }
        v_endif;
#endif
    }
    v_endif;
}
#endif

// The activation policy is mathematical: it applies after the compiler-bound
// target ingress transform above. Codegen supplies class codes from that
// target record, never from an activation name.
#if defined(TT_SPECIAL_VALUE_POLICY) &&                                                                \
    (defined(TT_TARGET_BH_BF16_SPECIAL_FINALIZER) || defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT) ||  \
     defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT) || defined(TT_TARGET_BH_BF16_RAW_NEG_INF_RESULT) || \
     defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT))

#ifndef TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT
#define TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT 0
#endif

static_assert(
    TT_TARGET_BH_BF16_SPECIAL_COMPARE_COUNT <= 3,
    "BH BF16 target special finalizer has invalid structural compare count");
template <int CODE>
inline vFloat target_special_class_value(vFloat computed, float constant = 0.0f) {
    static_assert(CODE >= 0 && CODE <= 6, "unsupported target special class");
    if constexpr (CODE == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    } else if constexpr (CODE == 1) {
        return std::numeric_limits<float>::infinity();
    } else if constexpr (CODE == 2) {
        return -std::numeric_limits<float>::infinity();
    } else if constexpr (CODE == 3) {
        return vFloat(0.0f);
    } else if constexpr (CODE == 4) {
        return setsgn(vFloat(0.0f), 1);
    } else if constexpr (CODE == 5) {
        return computed;  // finite_other is owned by the selected evaluator.
    } else {
        return vFloat(constant);
    }
}

#if defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF) || defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO)
#if defined(TT_SPECIAL_COMPARE_POS_INF) || defined(TT_SPECIAL_COMPARE_NEG_INF) || \
    defined(TT_SPECIAL_COMPARE_POS_ZERO) || defined(TT_TARGET_BH_BF16_ACTION_COORDINATE)
#error "magnitude finalizer arms exclude the per-class compare schedule"
#endif
inline void apply_target_special_policy(vFloat x_effective, vFloat& result) {
    // Merged-magnitude quotient: codegen emits these arms only when the
    // finalizer owns exactly {+Inf, -Inf, +0} effective ingress and the
    // egress narrowing maps the +Inf-class and -Inf-class results to one
    // output class, so both signed infinities may share the +Inf-policy
    // value.  Raw NaN of either sign has already canonicalized to +Inf at
    // the common SFPU DST load; raw -0 and both subnormal signs are exactly
    // the open MIN_NORMAL magnitude band, so no DAZ coordinate is needed.
    static_assert(
        TT_SPECIAL_POS_INF != 5 && TT_SPECIAL_POS_ZERO != 5,
        "magnitude finalizer requires direct structural class values");
    vFloat magnitude = setsgn(x_effective, 0);
#if defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_INF)
    {
        constexpr float bf16_max_finite = 3.3895313892515355e+38f;
        v_if(magnitude > bf16_max_finite) { result = target_special_class_value<TT_SPECIAL_POS_INF>(result); }
        v_endif;
    }
#endif
#if defined(TT_SPECIAL_COMPARE_MAGNITUDE_POS_ZERO)
    {
        constexpr float min_normal = std::numeric_limits<float>::min();
        v_if(magnitude < min_normal) { result = target_special_class_value<TT_SPECIAL_POS_ZERO>(result); }
        v_endif;
    }
#endif
}
#else
inline void apply_target_special_policy(vFloat x_effective, vFloat& result) {
    // Only the three classes reachable through BH BF16 ingress are tested.
    // Both raw NaN signs share the +Inf arm on the common BH SFPU path; raw -0
    // and both subnormal signs share the +0 arm. Apply mathematical policy to
    // that exact target quotient.
#if defined(TT_SPECIAL_COMPARE_POS_INF)
    if constexpr (TT_SPECIAL_POS_INF != 5) {
        constexpr float bf16_max_finite = 3.3895313892515355e+38f;
        vFloat positive_special_coordinate = x_effective;
        v_if(positive_special_coordinate > bf16_max_finite) {
            result = target_special_class_value<TT_SPECIAL_POS_INF>(result);
        }
        v_endif;
    }
#endif
#if defined(TT_SPECIAL_COMPARE_NEG_INF)
    if constexpr (TT_SPECIAL_NEG_INF != 5) {
        constexpr float bf16_min_finite = -3.3895313892515355e+38f;
        v_if(x_effective < bf16_min_finite) { result = target_special_class_value<TT_SPECIAL_NEG_INF>(result); }
        v_endif;
    }
#endif
#if defined(TT_SPECIAL_COMPARE_POS_ZERO)
    if constexpr (TT_SPECIAL_POS_ZERO != 5) {
        v_if(is_zero(x_effective)) { result = target_special_class_value<TT_SPECIAL_POS_ZERO>(result); }
        v_endif;
    }
#endif
}
#endif  // merged-magnitude vs per-class compare schedule

#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR)
inline void apply_target_raw_nan_policy(vUInt raw_u16, vFloat& result) {
    // DataLayout::U16 exposes sign in bit 15, mantissa in bits 14:8, and the
    // complete exponent byte in bits 7:0.
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) {
#if TT_SPECIAL_NAN == 5
            // ``finite_other`` is a class declaration, not a recoverable
            // value. No admitted direct structural lowering may invent one.
#error "raw NaN finite-other policy has no direct structural value"
#else
            result = target_special_class_value<TT_SPECIAL_NAN>(result);
#endif
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) && !defined(TT_TARGET_BH_BF16_RAW_POS_NAN_RESULT)
inline void apply_target_raw_pos_nan_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x80ffu);
    v_if(exponent_delta == 0u) {
        v_if((raw_u16 & vUInt(0x8000u)) == 0u) {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) {
#if TT_SPECIAL_NAN == 5
#error "raw positive NaN finite-other policy has no direct structural value"
#else
                result = target_special_class_value<TT_SPECIAL_NAN>(result);
#endif
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) && !defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_RESULT)
inline void apply_target_raw_neg_nan_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
    v_if(exponent_delta == 0u) {
        v_if((raw_u16 & vUInt(0x8000u)) != 0u) {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) {
#if TT_SPECIAL_NAN == 5
#error "raw negative NaN finite-other policy has no direct structural value"
#else
                result = target_special_class_value<TT_SPECIAL_NAN>(result);
#endif
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER)
inline void apply_target_raw_signed_nan_policy(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) {
            v_if((raw_u16 & vUInt(0x8000u)) == 0u) { result = target_special_class_value<TT_SPECIAL_POS_INF>(result); }
            v_else { result = target_special_class_value<TT_SPECIAL_NEG_INF>(result); }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}
#endif

#else
inline void apply_target_special_policy(vFloat x_effective, vFloat& result) {
    (void)x_effective;
    (void)result;
}
#endif

// Single-pass evaluator integration point.  Every evaluator calls this while
// its raw-coordinate vector is still live, after reconstruction/postcompose
// and before precision conversion, gradient multiplication, or store.  The
// no-action arm compiles away, preserving legacy generated source behavior.
inline void finalize_raw_domain_actions(vFloat x_raw, vFloat& result) {
    vFloat x_effective = target_domain_action_coordinate(x_raw);
#if defined(TT_SELECTED_CORE_AGGREGATE_STIRLING)
    apply_selected_core_aggregate_stirling(x_raw, result);
#elif defined(TT_SELECTED_CORE_TOTAL_FORM)
    apply_selected_core_total_form(x_effective, result);
#endif
#if defined(TT_DOMAIN_ACTION_PROGRAM)
    apply_raw_pre_domain_actions(x_effective, result);
#if defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP)
    result = sfpi::max(result, TT_DOMAIN_ACTION_LOWER_TERMINAL_VALUE);
#endif
#else
    (void)x_raw;
    (void)result;
#endif
    // Ordered finite/domain return_class actions own their lanes first.  The
    // effective special policy then has final class authority; target rounding
    // and FTZ happen in the evaluator's existing conversion/store epilogue.
#if !defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
    apply_target_special_policy(x_effective, result);
#endif
#if defined(TT_TARGET_BH_FP32_RAW_NEG_ZERO_DISCRIMINATOR)
    // FP32 ingress preserves the source bits.  Restore -0 after affine/clamp
    // arithmetic, which may otherwise canonicalize 1*(-0)+0 to +0.
    v_if(as<vUInt>(x_raw) == vUInt(0x80000000u)) { result = setsgn(vFloat(0.0f), 1); }
    v_endif;
#endif
}

#if defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
// Some BF16 contracts require classes that Blackhole's ordinary FP32->BF16
// conversion does not preserve (notably qNaN and -0).  The selected evaluator
// performs its normal narrowing first, then installs the typed class result
// from the still-live raw coordinate without narrowing it a second time.
inline void finalize_post_round_raw_domain_actions(vFloat x_raw, vFloat& result) {
    apply_target_special_policy(target_domain_action_coordinate(x_raw), result);
}
#endif

#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR) ||     \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT) ||                                                          \
    defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR) ||                                                     \
    defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR) ||   \
    defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER) || defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR) ||    \
    defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR) || defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR) ||                                                           \
    defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) ||                                                     \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
#define TT_TARGET_BH_BF16_HAS_ENCODED_RAW_TERMINAL 1
inline void finalize_encoded_raw_domain_actions(vUInt raw_u16, vFloat& result) {
#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NONFINITE_SPLIT)
    apply_target_raw_signed_nonfinite_split(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NAN_DISCRIMINATOR)
    apply_target_raw_nan_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_NAN_DISCRIMINATOR)
    apply_target_raw_pos_nan_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_NAN_DISCRIMINATOR)
    apply_target_raw_neg_nan_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_SIGNED_NAN_FINALIZER)
    apply_target_raw_signed_nan_policy(raw_u16, result);
#endif
// Compiled raw-class actions are the terminal overlay and therefore run after
// the target ingress quotient's generic NaN repair. Their exact predicates
// are disjoint from every remaining individual action below.
#if defined(TT_TARGET_BH_BF16_RAW_NONFINITE_DISCRIMINATOR)
    apply_target_raw_nonfinite_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_NONFINITE_DISCRIMINATOR)
    apply_target_raw_pos_nonfinite_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_DISCRIMINATOR)
    // U16 DST bytes are swapped relative to the IEEE BF16 word: -0 is 0x0080.
    v_if(raw_u16 == vUInt(0x0080u)) {
#if defined(TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT)
        result = target_raw_terminal_value<TT_TARGET_BH_BF16_RAW_NEG_ZERO_RESULT>(
            result, TT_TARGET_BH_BF16_RAW_NEG_ZERO_CONSTANT);
#else
        result = setsgn(vFloat(0.0f), 1);
#endif
    }
    v_endif;
#endif
#if defined(TT_TARGET_BH_BF16_RAW_NEG_INF_DISCRIMINATOR)
    apply_target_raw_neg_inf_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_INF_DISCRIMINATOR)
    apply_target_raw_pos_inf_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_ZERO_DISCRIMINATOR)
    apply_target_raw_pos_zero_policy(raw_u16, result);
#endif
#if defined(TT_TARGET_BH_BF16_RAW_POS_SUBNORMAL_DISCRIMINATOR) || \
    defined(TT_TARGET_BH_BF16_RAW_NEG_SUBNORMAL_DISCRIMINATOR)
    apply_target_raw_subnormal_policy(raw_u16, result);
#endif
}

inline void finalize_encoded_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    (void)x_raw;
    finalize_encoded_raw_domain_actions(raw_u16, result);
}

#if defined(TT_TARGET_BH_BF16_POST_ROUND_RAW_CLASS_REPAIR)
inline void finalize_post_round_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    finalize_post_round_raw_domain_actions(x_raw, result);
    finalize_encoded_raw_domain_actions(raw_u16, result);
}
#endif

inline void finalize_raw_domain_actions(vFloat x_raw, vUInt raw_u16, vFloat& result) {
    finalize_raw_domain_actions(x_raw, result);
    finalize_encoded_raw_domain_actions(x_raw, raw_u16, result);
}
#endif
