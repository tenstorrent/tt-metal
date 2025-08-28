// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_elu.h"
#include "llk_defs.h"
namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    _calculate_elu_<APPROX_MODE, ITERATIONS>(slope);
}

template <ApproximationMode APPROX_MODE>
void elu_init() {
    _init_elu_<APPROX_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
