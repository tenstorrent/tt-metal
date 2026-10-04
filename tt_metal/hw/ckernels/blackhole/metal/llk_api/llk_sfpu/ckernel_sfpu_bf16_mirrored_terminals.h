// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
// Include inside namespace sfpi. Typed callers retain the domain-action proof.

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

// A NaN of either sign takes one class: its sign is not information.
template <int Code>
inline void nan_class_terminal(vUInt raw_u16, vFloat& result) {
    vUInt exponent_delta = (raw_u16 ^ vUInt(0x00ffu)) & vUInt(0x00ffu);
    v_if(exponent_delta == 0u) {
        vUInt mantissa = raw_u16 & vUInt(0x7f00u);
        v_if(mantissa != 0u) { result = target_raw_terminal_value<Code>(result); }
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
