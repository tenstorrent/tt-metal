// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    [[maybe_unused]] bool EN_32BIT_DEST,
    int ITERATIONS = SFPU_ITERATIONS,
    [[maybe_unused]] bool legacy_compat = true>
inline void calculate_reciprocal() {
    static_assert(legacy_compat == true, "Non-default legacy_compat (false) not supported in Quasar reciprocal");
    _calculate_reciprocal_<APPROXIMATION_MODE, ITERATIONS>();
}

template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    [[maybe_unused]] bool EN_32BIT_DEST,
    [[maybe_unused]] bool legacy_compat = true>
void recip_init() {
    // Kept for backwards compatibility
    static_assert(legacy_compat == true, "Non-default legacy_compat (false) not supported in Quasar reciprocal");
}

}  // namespace sfpu
}  // namespace ckernel
