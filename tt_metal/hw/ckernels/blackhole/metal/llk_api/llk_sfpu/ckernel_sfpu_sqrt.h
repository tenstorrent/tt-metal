// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "llk_defs.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8, int RECIPROCAL_ITERATIONS = 2>
inline void calculate_sqrt() {
    _calculate_sqrt_<APPROX_MODE, ITERATIONS, RECIPROCAL_ITERATIONS>(ITERATIONS);
}

template <ApproximationMode APPROX_MODE>
void sqrt_init() {
    _init_sqrt_<APPROX_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
