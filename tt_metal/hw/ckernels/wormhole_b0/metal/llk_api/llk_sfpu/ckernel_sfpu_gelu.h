// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "llk_defs.h"

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE>
void gelu_init() {
    _init_gelu_<(APPROX_MODE == ApproximationMode::Fast)>();
}

template <ApproximationMode APPROX_MODE>
void gelu_derivative_init() {
    _init_gelu_derivative_<(APPROX_MODE == ApproximationMode::Fast)>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROX_MODE == ApproximationMode::Fast) {
        _calculate_gelu_<APPROX_MODE, ITERATIONS>();
    } else {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat result = in;
        v_if(in == 0.0f) { result = 0.0f; }
        v_elseif(in < 3.0f) { result = calculate_gelu_chebyshev(in); }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
    }
}

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    _calculate_gelu_derivative_<(APPROX_MODE == ApproximationMode::Fast), ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
