// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ckernel_sfpu_bf16_sfpi_isa.h"
// Include inside namespace sfpi.

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

template <int Code>
inline void signed_nonfinite_split_terminal(vUInt raw_u16, vFloat& result, float constant) {
    // One exponent-FF test serves the positive constant and the negative-NaN
    // class. Negative infinity is left to the numeric body; a nonzero negative
    // mantissa is a NaN.
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
