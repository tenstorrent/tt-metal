// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_sqrt.h"

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    [[maybe_unused]] bool EN_32BIT_DEST,
    [[maybe_unused]] bool FAST_APPROX = false>
inline void calculate_sqrt() {
    static_assert(FAST_APPROX == false, "Non-default FAST_APPROX (true) not supported in Quasar sqrt");
    _calculate_sqrt_<APPROXIMATION_MODE, ITERATIONS>();
}

template <[[maybe_unused]] bool APPROXIMATION_MODE>
void sqrt_init() {
    // Empty function kept for backwards compatibility
}

}  // namespace sfpu
}  // namespace ckernel
