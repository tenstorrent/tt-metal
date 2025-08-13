// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_welfords_sfpu2.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {
ALWI void welford(uint32_t dst0, uint32_t dst1, uint32_t dst3, uint32_t start_N, uint32_t end_N, uint32_t last_run) {
    MATH(llk_math_welfords_sfpu(dst0, dst1, dst3, start_N, end_N, last_run));
}

/**
 * Uses a copy of the ternery_sfpu_init
 */
// EXPERIMENTAL
ALWI void welford_init() { MATH((llk_math_welfords_sfpu_init())); }
}  // namespace ckernel
