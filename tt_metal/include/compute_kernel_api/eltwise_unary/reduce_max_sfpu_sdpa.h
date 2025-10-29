// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_sfpu_reduce_sdpa.h"
#endif

namespace ckernel {

template <bool fast_and_approx = false>
ALWI void sfpu_reduce_max_sdpa_init() {
    MATH((llk_math_sfpu_reduce_max_sdpa_init<fast_and_approx>()));
}

template <bool fast_and_approx = false>
ALWI void sfpu_reduce_max_sdpa(uint32_t idst) {
    MATH((llk_math_sfpu_reduce_max_sdpa<fast_and_approx>(idst)));
}

}  // namespace ckernel
