// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_dropout.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_dropout(uint probability, uint scale) {
    _calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, probability, scale);
}

template <bool APPROXIMATION_MODE>
inline void dropout_init(const uint seed) {
    _init_dropout_(seed);
}

}  // namespace sfpu
}  // namespace ckernel
