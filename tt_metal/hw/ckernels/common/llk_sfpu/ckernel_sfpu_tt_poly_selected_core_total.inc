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
#if defined(TT_WH_SYMMETRIC_EXP_POOL)
    static_assert(TT_SELECTED_CORE_EXP_DEGREE == 3u && TT_SELECTED_CORE_EXP_COEFFS[0] == 1.0f);
    vFloat exponential = vConstFloatPrgm0 * fraction + vConstFloatPrgm1;
    exponential = exponential * fraction + vConstFloatPrgm2;
    exponential = exponential * fraction + 1.0f;
#else
    vFloat exponential = selected_core_total_poly<TT_SELECTED_CORE_EXP_DEGREE>(TT_SELECTED_CORE_EXP_COEFFS, fraction);
#endif
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
#include "symmetric_factored_log.h"
inline vFloat selected_core_total_direct_log(vFloat x) {
    static_assert(TT_SELECTED_CORE_LOG_DEGREE == 3u);
    vFloat result = selected_direct_log_core(x, TT_SELECTED_CORE_LOG_COEFFS);
#if !defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1)
    v_if(is_nan(x) || is_inf(x)) { result = setman(x, 0); }
    v_endif;
#endif
    return result;
}
#endif

#if defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
#include "core_bridge_exponent.inc"
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
#if defined(ARCH_WORMHOLE)
        // The tail excludes zero and nonfinite inputs.  Testing x directly
        // releases magnitude before WH's four-register reciprocal.
        v_if(x < 0.0f) { result = negative_factor; }
#else
        vFloat nonnegative_selector = x + magnitude;
        v_if(is_zero(nonnegative_selector)) { result = negative_factor; }
#endif
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
    selected_core_bridge_finalize(x, result);
#elif defined(TT_SELECTED_CORE_SYMMETRIC_DIRECT_LOG_TAIL_1)
    symmetric_direct_log_tail<__builtin_bit_cast(uint32_t, TT_SELECTED_CORE_LOWER),
        __builtin_bit_cast(uint32_t, TT_SELECTED_CORE_UPPER),
        __builtin_bit_cast(uint32_t, TT_SELECTED_CORE_TAIL_ADDEND)>(x, result, TT_SELECTED_CORE_LOG_COEFFS);
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
