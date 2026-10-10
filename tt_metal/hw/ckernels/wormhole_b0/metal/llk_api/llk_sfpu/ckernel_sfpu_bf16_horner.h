// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "sfpi.h"

namespace sfpi {

// Compile-time load indices are required by DST's immediate-row SFPLOAD.
// A pointer/Config accessor and a parked-DST provider use the same rungs.
template <typename Coefficients>
struct polynomial_indexed_coefficients {
    const Coefficients& values;
    template <uint32_t INDEX>
    inline vFloat load() const {
        return values[INDEX];
    }
};

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

template <uint32_t DEGREE, typename Coefficients>
inline void eval_polynomial_dual(const Coefficients& coeffs, vFloat x1, vFloat x2, vFloat& result1, vFloat& result2) {
    static_assert(DEGREE <= 16, "unsupported canonical paired polynomial degree");
    if constexpr (DEGREE == 1) {
        // Retain the canonical immediate path's expression and load order.
        // The DST path above deliberately keeps accumulator*x+coefficient.
        vFloat c1 = coeffs[1];
        vFloat c0 = coeffs[0];
        result1 = c0 + c1 * x1;
        result2 = c0 + c1 * x2;
    } else {
        eval_polynomial_transport<DEGREE>(
            polynomial_indexed_coefficients<Coefficients>{coeffs}, x1, x2, result1, result2);
    }
}

}  // namespace sfpi
