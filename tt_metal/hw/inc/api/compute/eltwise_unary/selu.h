// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_selu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void selu_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_selu, RC, APPROX, idst)); }

ALWI void selu_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(selu, ckernel::sfpu::selu_init, APPROX)); }

}  // namespace ckernel
