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
    _init_gelu_<(APPROX_MODE)>();
}

template <ApproximationMode APPROX_MODE>
void gelu_derivative_init() {
    _init_gelu_derivative_<(APPROX_MODE)>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_gelu() {
    _calculate_gelu_<(APPROX_MODE), ITERATIONS>();
}

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_gelu_derivative() {
    _calculate_gelu_derivative_<(APPROX_MODE), ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
