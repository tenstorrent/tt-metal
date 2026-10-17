// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
// Include inside namespace sfpi, after the canonical polynomial Horner helper.
template <uint32_t Degree, bool MirrorFold>
inline void symmetric_factored_core_dual(
    const float* negative, const float* positive, float split, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    static_assert(Degree == 8);
    if constexpr (MirrorFold) {
        eval_polynomial_dual<Degree>(positive, setsgn(x1, 0), setsgn(x2, 0), result1, result2);
    } else {
        eval_polynomial_dual<Degree>(negative, x1, x2, result1, result2);
        vFloat tmp1, tmp2;
        eval_polynomial_dual<Degree>(positive, x1, x2, tmp1, tmp2);
        vFloat boundary = split;
        v_if(x1 >= boundary) { result1 = tmp1; }
        v_endif;
        v_if(x2 >= boundary) { result2 = tmp2; }
        v_endif;
    }
}

inline vFloat selected_direct_log_core(vFloat x, const float* coefficients) {
    vInt exponent = as<vInt>(vFloat(0.75f));
    exponent = as<vInt>(x) - exponent;
    exponent = as<vInt>(setman(as<vFloat>(exponent), 0));
    vFloat residual = as<vFloat>(as<vInt>(x) - exponent) - 1.0f;
    vFloat result = coefficients[3];
    result = result * residual + coefficients[2];
    result = result * residual + coefficients[1];
    result = result * residual + coefficients[0];
    result = residual + (residual * residual) * result;
    vFloat exponent_float = convert<vFloat>(abs(exponent), RoundMode::Nearest);
    exponent_float = copysgn(exponent_float, as<vFloat>(exponent));
    result = exponent_float * 0x1.62e43p-24f + result;
    return result;
}

template <uint32_t LowerBits, uint32_t UpperBits, uint32_t AddendBits>
inline void symmetric_direct_log_tail(vFloat x, vFloat& result, const float* coefficients) {
    v_if(is_inf(x)) { result = x; }
    v_elseif(x < __builtin_bit_cast(float, LowerBits) || x > __builtin_bit_cast(float, UpperBits)) {
        vFloat magnitude = setsgn(x, 0);
        magnitude = selected_direct_log_core(magnitude, coefficients) + __builtin_bit_cast(float, AddendBits);
        result = copysgn(magnitude, x);
    }
    v_endif;
}
