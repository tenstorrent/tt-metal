// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "sfpi.h"

namespace ckernel::sfpu::ttpoly::acos {
using namespace ::sfpi;
using ::sfpi::DataLayout;
__attribute__((always_inline)) inline vFloat sfpu_mad(vFloat a, vFloat b, vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), SFPMAD_MOD1_OFFSET_NONE);
}

template <uint32_t DEGREE, uint32_t INDEX, typename Coefficients>
__attribute__((always_inline)) inline vFloat polynomial_horner_step(const Coefficients& coeffs, vFloat x) {
    if constexpr (INDEX == DEGREE) {
        return coeffs[INDEX];
    } else {
        vFloat accumulator = polynomial_horner_step<DEGREE, INDEX + 1>(coeffs, x);
        return sfpu_mad(accumulator, x, coeffs[INDEX]);
    }
}

template <uint32_t DEGREE, typename Coefficients>
__attribute__((always_inline)) inline vFloat eval_polynomial(const Coefficients& coeffs, vFloat x) {
    static_assert(DEGREE <= 16 || DEGREE == 32, "unsupported canonical polynomial degree");
    return polynomial_horner_step<DEGREE, 0>(coeffs, x);
}

}  // namespace ckernel::sfpu::ttpoly::acos
