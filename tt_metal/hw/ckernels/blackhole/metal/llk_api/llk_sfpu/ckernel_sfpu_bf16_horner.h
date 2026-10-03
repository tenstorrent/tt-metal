// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "sfpi.h"

namespace sfpi {

template <int INDEX, typename Coefficients>
__attribute__((always_inline)) inline void polynomial_transport_rungs(
    const Coefficients& coeffs, vFloat x, vFloat& result) {
    if constexpr (INDEX >= 0) {
        result = result * x + coeffs.template load<static_cast<uint32_t>(INDEX)>();
        polynomial_transport_rungs<INDEX - 1>(coeffs, x, result);
    }
}

template <int INDEX, typename Coefficients>
__attribute__((always_inline)) inline void polynomial_transport_rungs(
    const Coefficients& coeffs, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    if constexpr (INDEX >= 0) {
        vFloat c = coeffs.template load<static_cast<uint32_t>(INDEX)>();
        result1 = result1 * x1 + c;
        result2 = result2 * x2 + c;
        polynomial_transport_rungs<INDEX - 1>(coeffs, x1, x2, result1, result2);
    }
}

template <uint32_t DEGREE, typename Coefficients>
__attribute__((always_inline)) inline void eval_polynomial_transport(
    const Coefficients& coeffs, vFloat x, vFloat& result) {
    result = coeffs.template load<DEGREE>();
    polynomial_transport_rungs<static_cast<int>(DEGREE) - 1>(coeffs, x, result);
}

template <uint32_t DEGREE, typename Coefficients>
__attribute__((always_inline)) inline void eval_polynomial_transport(
    const Coefficients& coeffs, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    vFloat c = coeffs.template load<DEGREE>();
    result1 = c;
    result2 = c;
    polynomial_transport_rungs<static_cast<int>(DEGREE) - 1>(coeffs, x1, x2, result1, result2);
}

}  // namespace sfpi
