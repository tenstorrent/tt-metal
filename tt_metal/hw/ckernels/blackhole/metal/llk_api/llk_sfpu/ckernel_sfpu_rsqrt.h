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
inline void calculate_rsqrt() {
    _calculate_rsqrt_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS);
}

template <bool APPROXIMATION_MODE>
void rsqrt_init() {
    _init_rsqrt_<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
