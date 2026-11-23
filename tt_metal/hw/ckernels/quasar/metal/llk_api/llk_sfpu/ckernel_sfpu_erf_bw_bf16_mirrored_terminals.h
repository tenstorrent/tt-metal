// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu::ttpoly::erf_bw {
using namespace ::sfpi;
using ::sfpi::DataLayout;
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
inline void negative_infinity_terminal(vFloat input, vFloat& result) {
    if constexpr (Code != 5) {
        constexpr float bf16_min_finite = -3.3895313892515355e+38f;
        v_if(input < bf16_min_finite) { result = target_raw_terminal_value<Code>(result); }
        v_endif;
    }
}

// Quasar: the NaN classes are read from the float value of x (sign, exponent 255, nonzero mantissa),
// not from Blackhole's swizzled raw 16-bit Dest bits, and with signed compares only: an unsigned
// compare against an immediate lowers to an SFPIADD into read-only LREG8.
template <int Code, bool Nested>
inline void negative_nan_class_terminal(vFloat x_raw, vFloat& result) {
    static_assert(Code >= 0 && Code <= 4, "negative NaN requires a class result");
    if constexpr (Nested) {
        v_if(is_neg(x_raw) && exexp(x_raw, ExponentMode::Biased) >= 255) {
            v_if(exman(x_raw) != 0) { result = target_raw_terminal_value<Code>(result); }
            v_endif;
        }
        v_endif;
    } else {
        v_if(is_neg(x_raw) && exexp(x_raw, ExponentMode::Biased) >= 255 && exman(x_raw) != 0) {
            result = target_raw_terminal_value<Code>(result);
        }
        v_endif;
    }
}

template <int Code>
inline void positive_nan_class_terminal(vFloat x_raw, vFloat& result, float constant = 0.0f) {
    static_assert(Code >= 0 && Code <= 6);
    v_if(is_pos(x_raw) && exexp(x_raw, ExponentMode::Biased) >= 255) {
        v_if(exman(x_raw) != 0) { result = target_raw_terminal_value<Code>(result, constant); }
        v_endif;
    }
    v_endif;
}
}  // namespace ckernel::sfpu::ttpoly::erf_bw
