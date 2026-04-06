// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_softshrink.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void softshrink_tile(uint32_t idst, uint32_t param0) {
    MATH(SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_softshrink, RC, APPROX, idst, param0));
}

ALWI void softshrink_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(softshrink, APPROX)); }

}  // namespace ckernel
