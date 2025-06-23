// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
void gelu_init() {
    _init_gelu_<APPROXIMATION_MODE>();
}

template <bool APPROXIMATION_MODE>
void gelu_derivative_init() {
    _init_gelu_derivative_<APPROXIMATION_MODE>();
}

template <int ITERATIONS>
inline void calculate_gelu_appx() {
    constexpr bool approximation_mode = true;
    _calculate_gelu_<approximation_mode, ITERATIONS>(ITERATIONS);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_gelu_appx<ITERATIONS>();
    } else {
        // SFPU microcode
        _calculate_gelu_accurate_(ITERATIONS);
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    _calculate_gelu_derivative_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS);
}

}  // namespace sfpu
}  // namespace ckernel
