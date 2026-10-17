// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "sfpi.h"

namespace sfpi {

inline vFloat sfpu_mad(vFloat a, vFloat b, vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), SFPMAD_MOD1_OFFSET_NONE);
}

// Coefficients may be the canonical LUT pointer or a generated Config accessor.
// Both retain the descending, single-rounded MAD chain of the canonical body.
template <uint32_t DEGREE, uint32_t INDEX, typename Coefficients>
inline vFloat polynomial_horner_step(const Coefficients& coeffs, vFloat x) {
    if constexpr (INDEX == DEGREE) {
        return coeffs[INDEX];
    } else {
        vFloat accumulator = polynomial_horner_step<DEGREE, INDEX + 1>(coeffs, x);
        return sfpu_mad(accumulator, x, coeffs[INDEX]);
    }
}

template <uint32_t DEGREE, typename Coefficients>
inline vFloat eval_polynomial(const Coefficients& coeffs, vFloat x) {
    static_assert(DEGREE <= 16 || DEGREE == 32, "unsupported canonical polynomial degree");
    if constexpr (DEGREE == 32) {
        // Preserve the canonical high-degree loop rather than introducing a
        // new unrolling policy into an existing selected schedule.
        vFloat result = coeffs[32];
        for (int i = 31; i >= 0; --i) {
            result = sfpu_mad(result, x, coeffs[i]);
        }
        return result;
    } else {
        return polynomial_horner_step<DEGREE, 0>(coeffs, x);
    }
}

}  // namespace sfpi
