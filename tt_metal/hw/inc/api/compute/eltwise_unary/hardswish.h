// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/eltwise_unary/eltwise_unary.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_hardswish.h"
#endif

namespace ckernel {

ALWI void hardswish_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(hardswish, false)); }

ALWI void hardswish_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_hardswish, RC, false, idst)); }

}  // namespace ckernel
