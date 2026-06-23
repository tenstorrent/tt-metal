// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_gelu.h"

namespace ckernel {
namespace sfpu {

template <[[maybe_unused]] bool APPROXIMATION_MODE>
inline void gelu_init() {
    _init_gelu_();
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_gelu() {
    _calculate_gelu_<ITERATIONS>();
}

}  // namespace sfpu
}  // namespace ckernel
