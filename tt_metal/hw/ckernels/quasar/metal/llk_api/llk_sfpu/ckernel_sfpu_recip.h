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
    int ITERATIONS = 8,
    [[maybe_unused]] bool legacy_compat = false>
inline void calculate_reciprocal() {
    static_assert(EN_32BIT_DEST == false, "Non-default EN_32BIT_DEST not supported in quasar reciprocal");
    static_assert(legacy_compat == false, "Non-default legacy_compat not supported in quasar reciprocal");
    _calculate_reciprocal_<APPROXIMATION_MODE>(ITERATIONS);
}

template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    [[maybe_unused]] bool EN_32BIT_DEST,
    [[maybe_unused]] bool legacy_compat = false>
void recip_init() {
    // Kept for backwards compatibility
    static_assert(EN_32BIT_DEST == false, "Non-default EN_32BIT_DEST not supported in quasar reciprocal");
    // Not sure if we are actually supposed to assert on this
    // Seems like it is used to control whether specific constants are programmed
    // As part of initialization, hence it's sometimes true and sometimes false
    // Also the default in recip_tile_init() is true but for recip_init() its false (from BH)
    static_assert(legacy_compat == false, "Non-default legacy_compat not supported in quasar reciprocal");
}

}  // namespace sfpu
}  // namespace ckernel
