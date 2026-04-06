// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_lgamma.h"
#endif

namespace ckernel {

ALWI void lgamma_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROX>())); }

ALWI void lgamma_tile(uint32_t idst) { MATH((SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_lgamma, RC, APPROX, idst))); }

}  // namespace ckernel
