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
ALWI void welford(
    uint32_t dst0,
    uint32_t dst1,
    uint32_t dst2,
    uint32_t current_sample,
    uint32_t final_sample,
    uint32_t reformat_dst = 1,
    uint32_t skip_n_samples = 0) {
    MATH(llk_math_welfords_sfpu(dst0, dst1, dst2, current_sample, final_sample, reformat_dst, skip_n_samples));
}

/**
 * Uses a copy of the ternery_sfpu_init
 */
// EXPERIMENTAL
ALWI void welford_init() { MATH((llk_math_welfords_sfpu_init())); }
}  // namespace ckernel
