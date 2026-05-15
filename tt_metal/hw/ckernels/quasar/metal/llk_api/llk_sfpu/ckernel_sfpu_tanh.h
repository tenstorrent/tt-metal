// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_tanh.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_tanh(const int iterations) {
    _calculate_tanh_<APPROXIMATION_MODE>(iterations);
}

}  // namespace sfpu
}  // namespace ckernel
