// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_digamma.h"
#endif

namespace ckernel {

ALWI void digamma_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_digamma, RC, APPROX, idst)); }

ALWI void digamma_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(unused, APPROX)); }

}  // namespace ckernel
