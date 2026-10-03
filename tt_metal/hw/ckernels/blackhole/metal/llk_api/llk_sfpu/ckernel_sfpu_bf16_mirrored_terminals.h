// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
