// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_elu.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    _calculate_elu_<APPROXIMATION_MODE, ITERATIONS>(slope);
}

template <bool APPROXIMATION_MODE>
void elu_init() {
    _init_elu_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
