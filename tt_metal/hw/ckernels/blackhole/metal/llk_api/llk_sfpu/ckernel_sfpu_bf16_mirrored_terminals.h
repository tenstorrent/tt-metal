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

// Existing effective-class negative-infinity finalizer.
template <int Code>
inline void negative_infinity_terminal(vFloat input, vFloat& result) {
    if constexpr (Code != 5) {
        constexpr float bf16_min_finite = -3.3895313892515355e+38f;
        v_if(input < bf16_min_finite) { result = target_raw_terminal_value<Code>(result); }
        v_endif;
    }
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
