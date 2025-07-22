// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_int_sum.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void sfpu_sum_int_init() { MATH(SFPU_INIT_KERNEL_CALL(unused, sfpu::sum_int_init, APPROX)); }

ALWI void sfpu_sum_int_col(uint32_t idst) {
    MATH(SFPU_ONE_PARAM_KERNEL(calculate_sum_int_col, APPROX, idst, (int)VectorMode::R));
}

ALWI void sfpu_sum_int_row(uint32_t idst) {
    MATH(SFPU_ONE_PARAM_KERNEL(calculate_sum_int_row, APPROX, idst, (int)VectorMode::C));
}

ALWI void sfpu_add_int(uint32_t idst, uint32_t dst_offset = 2, int32_t iterations = 8) {
    MATH(SFPU_TWO_PARAM_KERNEL_ONE_RUNTIME(add_int, APPROX, 8, idst, (int)VectorMode::RC, dst_offset));
}

}  // namespace ckernel
