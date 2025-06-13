// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sqrt() {
    _calculate_sqrt_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS);
}

template <bool APPROXIMATION_MODE>
void sqrt_init() {
    _init_sqrt_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
