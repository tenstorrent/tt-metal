// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
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

#if defined(TT_SELECTED_WH_AGGREGATE_STIRLING)
// Public BF16 ELWSUB on this fold's lower interval (-128, 1.5].
// The declared half-integer shifts are exact on the larger operand's
// 11-bit grid. Round a smaller input onto that grid, then subtract and
// narrow with ties away. This is the ordinary FPU arithmetic, not a pole cut.
inline vFloat selected_public_fpu_subtract(vFloat x, uint32_t shift_bits) {
    if (shift_bits == 0u) { return x; }
    const int shift_exponent = (shift_bits >> 23) & 255;
    const float base = __builtin_bit_cast(float, uint32_t(shift_exponent << 23));
    const float scale = __builtin_bit_cast(float, uint32_t((264 - shift_exponent) << 23));
    const float inverse_scale = __builtin_bit_cast(float, uint32_t((shift_exponent - 10) << 23));
    vFloat argument = x;
    v_if(setsgn(x, 0) < base) {
        vFloat scaled = setsgn(x, 0) * scale + 1024.5f;
        vInt integer = as<vInt>(exman(scaled) >> 13);
        vSMag aligned = convert<vSMag>(integer);
        vFloat magnitude = convert<vFloat>(aligned, RoundMode::Nearest) * inverse_scale;
        argument = copysgn(magnitude, x);
    }
    v_endif;
    argument = argument - __builtin_bit_cast(float, shift_bits);
    vUInt bits = as<vUInt>(argument);
    return as<vFloat>((bits + vUInt(0x8000u)) & vUInt(0xffff0000u));
}
#endif

inline vInt selected_public_shifted_lgamma_any_nonfinite(vFloat x) {
    vInt any_nonfinite = 0;
#pragma GCC unroll 4
    for (uint32_t term = 0; term < TT_SELECTED_AGGREGATE_CLASS_TERM_COUNT; ++term) {
#if defined(TT_SELECTED_WH_AGGREGATE_STIRLING)
        vFloat argument = selected_public_fpu_subtract(
            x, TT_SELECTED_AGGREGATE_CLASS_SHIFT_BITS[term]);
#else
        vFloat shift = __builtin_bit_cast(float, TT_SELECTED_AGGREGATE_CLASS_SHIFT_BITS[term]);
        vFloat argument = x - shift;
        argument = convert<vFloat16b>(argument, RoundMode::Nearest);
#endif
        any_nonfinite |= selected_public_lgamma_lower_nonfinite(argument);
    }
#if !defined(TT_SELECTED_WH_AGGREGATE_STIRLING)
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
#endif
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

#if defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND)
// The ordinary WH floating pack path canonicalizes NaNs to infinity. This
// terminal supplies canonical BF16 words to the declared UInt16 output ABI;
// all finite results keep the same preceding target rounding.
inline vUInt mathematical_aggregate_bf16_word(int d, vFloat result) {
    vUInt output = as<vUInt>(result) >> 16;
    constexpr float lower = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_LOWER_BITS);
    vFloat source = dst_reg[d];
    v_if(source <= lower) { output = 0x7fc0u; }
    v_endif;
    vUInt raw = dst_reg[d].mode<DataLayout::U16>();
    v_if(((raw & vUInt(0x00ffu)) == vUInt(0x00ffu)) &&
         ((raw & vUInt(0x7f00u)) != vUInt(0u))) {
        output = 0x7fc0u;
    }
    v_endif;
    return output;
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

#if defined(TT_SELECTED_WH_AGGREGATE_STIRLING)
    constexpr float core_lower = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_LOWER_BITS);
    vInt source_exponent = exexp(setsgn(x, 0), ExponentMode::Biased);
    v_if(source_exponent == 255) {
        y = copysgn(vFloat(std::numeric_limits<float>::infinity()), x);
    }
    v_elseif(x <= -TT_SELECTED_AGGREGATE_TERM_OVERFLOW) {
        y = -std::numeric_limits<float>::infinity();
    }
    v_elseif(x <= -128.0f || is_zero(x)) {
        // Every finite BF16 value <= -128 is an exact negative integer.
        y = std::numeric_limits<float>::infinity();
    }
    v_elseif(x <= core_lower) {
        vInt any_nonfinite = selected_public_shifted_lgamma_any_nonfinite(x);
        vInt zero_representative = selected_public_fold_zero_representative(x);
        v_if(zero_representative != 0) { y = 0.0f; }
        v_elseif(any_nonfinite != 0) { y = std::numeric_limits<float>::infinity(); }
        v_endif;
    }
    v_endif;
#elif !defined(TT_SELECTED_AGGREGATE_MATHEMATICAL_POST_ROUND)
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
#endif
}
#endif

#if defined(TT_SELECTED_CORE_AGGREGATE_STIRLING)
inline vFloat selected_aggregate_stirling_prepare(vFloat x_raw) {
    constexpr float core_lower = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_LOWER_BITS);
    constexpr float core_upper = __builtin_bit_cast(float, TT_SELECTED_STIRLING_CORE_UPPER_BITS);
    vFloat prepared = x_raw;
    v_if(prepared < core_lower) { prepared = core_lower; }
    v_elseif(prepared > core_upper) { prepared = core_upper; }
    v_endif;
    return prepared;
}
#endif
