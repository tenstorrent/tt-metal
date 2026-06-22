// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_tanh.h"

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_tanh() {
    _calculate_tanh_<true /* APPROX */, ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
