// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Include inside namespace sfpi. Typed callers retain the domain-action proof.
template <uint32_t BoundBits, bool SignedEndpoints>
inline void mirrored_class_terminals(vFloat x_raw, vFloat& result) {
    vFloat absolute_raw = setsgn(x_raw, 0);
    if constexpr (SignedEndpoints) {
        v_if(absolute_raw > __builtin_bit_cast(float, BoundBits)) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_elseif(absolute_raw >= __builtin_bit_cast(float, BoundBits)) {
            vFloat endpoint_inf = std::numeric_limits<float>::infinity();
            result = copysgn(endpoint_inf, x_raw);
        }
        v_endif;
    } else {
        v_if(absolute_raw > __builtin_bit_cast(float, BoundBits)) { result = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
}

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

// Existing effective-class negative-infinity finalizer.
template <int Code>
inline void negative_infinity_terminal(vFloat input, vFloat& result) {
    if constexpr (Code != 5) {
        constexpr float bf16_min_finite = -3.3895313892515355e+38f;
        v_if(input < bf16_min_finite) { result = target_raw_terminal_value<Code>(result); }
        v_endif;
    }
}

// The existing unsigned BF16 transport predicate for either NaN sign.
template <int Code>
inline void nan_union_class_terminal(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = target_raw_terminal_value<Code>(result); }
        v_endif;
    }
    v_endif;
}

// Keep the selected BH conjunction and WH nested predicate distinct.
template <int Code, bool Nested>
inline void negative_nan_class_terminal(vUInt raw_u16, vFloat& result) {
    static_assert(Code >= 0 && Code <= 4, "negative NaN requires a class result");
    if constexpr (Nested) {
        vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x80ffu)) & vUInt(0x80ffu);
        v_if(exponent_and_sign_delta == 0u) {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) { result = target_raw_terminal_value<Code>(result); }
            v_endif;
        }
        v_endif;
    } else {
        vUInt exponent_and_sign = raw_u16 & vUInt(0x80ffu);
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(exponent_and_sign == vUInt(0x80ffu) && mantissa != 0u) {
            result = target_raw_terminal_value<Code>(result);
        }
        v_endif;
    }
}

template <int Code>
inline void positive_nan_class_terminal(vUInt raw_u16, vFloat& result, float constant = 0.0f) {
    static_assert(Code >= 0 && Code <= 6);
    vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x80ffu);
    v_if(exponent_and_sign_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = target_raw_terminal_value<Code>(result, constant); }
        v_endif;
    }
    v_endif;
}

template <int PositiveCode, int NegativeCode>
inline void signed_nan_class_terminal(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) {
            v_if((raw_u16 & vUInt(0x8000u)) == 0u) { result = target_raw_terminal_value<PositiveCode>(result); }
            v_else { result = target_raw_terminal_value<NegativeCode>(result); }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}

// Canonical ordered raw-action records. Product callers retain their selected
// early exits/interleaving and opt into the existing negative-tail narrowing.
template <typename Config, uint32_t INDEX, bool NegativeTailFold = false, typename Float>
inline void apply_raw_domain_record(Float x_raw, Float& result) {
    constexpr auto record = Config::kDomainActions[INDEX];
    static_assert(record.action_kind <= 4, "unsupported domain-action kind");
    static_assert(record.return_class <= 4, "unsupported domain-action return class");
    Float action_value = record.value;
    if constexpr (record.action_kind == 1) {
        action_value = x_raw;
    } else if constexpr (record.action_kind == 2) {
        Float action_scale = record.scale;
        Float action_bias = record.bias;
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
    if constexpr (NegativeTailFold && INDEX == 1) {
        action_value = x_raw * 0.0f;
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
}

template <typename Config, uint32_t INDEX>
inline void apply_raw_domain_records(vFloat x_raw, vFloat& result) {
    apply_raw_domain_record<Config, INDEX>(x_raw, result);
    if constexpr (INDEX > 0) {
        apply_raw_domain_records<Config, INDEX - 1>(x_raw, result);
    }
}

template <int Code>
inline void positive_nonfinite_terminal(vUInt raw_u16, vFloat& result, float constant) {
    // Physical BF16 layout has the complete exponent byte in bits 7:0 and
    // the sign in bit 15. This one quotient is exactly +Inf plus all 127
    // positive NaNs; codegen admits it only for equal typed constant actions.
    vUInt exponent_and_sign_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x80ffu);
    v_if(exponent_and_sign_delta == 0u) { result = target_raw_terminal_value<Code>(result, constant); }
    v_endif;
}

template <int Code>
inline void signed_nonfinite_split_terminal(vUInt raw_u16, vFloat& result, float constant) {
    // Share the physical exponent-FF test between the positive constant
    // quotient and signed-ingress negative-NaN policy. Negative infinity is
    // left to the numeric body; a nonzero negative mantissa is the NaN arm.
    vUInt exponent = raw_u16 & vUInt(0x00ffu);
    v_if(exponent == vUInt(0x00ffu)) {
        vUInt sign = raw_u16 & vUInt(0x8000u);
        v_if(sign == 0u) { result = target_raw_terminal_value<Code>(result, constant); }
        v_else {
            vUInt mantissa = raw_u16 & vUInt(0x7f00u);
            v_if(mantissa != 0u) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }
        v_endif;
    }
    v_endif;
}

inline vFloat raw_daz_action_coordinate(vFloat x_raw) {
    constexpr float min_normal = std::numeric_limits<float>::min();
    vFloat effective = x_raw;
    v_if(effective > -min_normal) { effective = 0.0f; }
    v_endif;
    v_if(x_raw >= min_normal) { effective = x_raw; }
    v_endif;
    return effective;
}
template <int Code>
inline void zero_class_terminal(vFloat input, vFloat& result) {
    if constexpr (Code != 5) {
        v_if(is_zero(input)) { result = target_raw_terminal_value<Code>(result); }
        v_endif;
    }
}
template <uint32_t Word, int Code>
inline void encoded_word_terminal(vUInt raw_u16, vFloat& result, float constant = 0.0f) {
    v_if(raw_u16 == vUInt(Word)) { result = target_raw_terminal_value<Code>(result, constant); }
    v_endif;
}
template <int PositiveCode, int NegativeCode>
inline void encoded_subnormal_terminal(
    vUInt raw_u16, vFloat& result, float positive_constant = 0.0f, float negative_constant = 0.0f) {
    vUInt exponent = raw_u16 & vUInt(0x00ffu);
    vUInt mantissa = raw_u16 & vUInt(0x7f00u);
    v_if((exponent == 0u) && (mantissa != 0u)) {
        vUInt sign = raw_u16 & vUInt(0x8000u);
        if constexpr (PositiveCode >= 0) {
            v_if(sign == 0u) { result = target_raw_terminal_value<PositiveCode>(result, positive_constant); }
            v_endif;
        }
        if constexpr (NegativeCode >= 0) {
            v_if(sign != 0u) { result = target_raw_terminal_value<NegativeCode>(result, negative_constant); }
            v_endif;
        }
    }
    v_endif;
}
