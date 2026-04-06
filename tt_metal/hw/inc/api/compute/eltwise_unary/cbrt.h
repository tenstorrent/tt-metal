// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_cbrt.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void cbrt_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_cbrt, RC, APPROX, idst)); }

ALWI void cbrt_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(cbrt, APPROX)); }

}  // namespace ckernel
