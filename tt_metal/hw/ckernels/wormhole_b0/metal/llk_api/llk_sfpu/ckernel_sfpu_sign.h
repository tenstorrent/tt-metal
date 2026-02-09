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
inline void calculate_sign(const uint exponent_size_8) {
    _calculate_sign_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, exponent_size_8);
}

}  // namespace sfpu
}  // namespace ckernel
