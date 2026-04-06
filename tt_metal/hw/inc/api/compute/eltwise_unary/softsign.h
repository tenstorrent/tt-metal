// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_softsign.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void softsign_tile(uint32_t idst) { MATH(SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_softsign, RC, APPROX, idst)); }

ALWI void softsign_tile_init() { MATH(SFPU_INIT_KERNEL_CALL(softsign, _init_softsign_, APPROX)); }

}  // namespace ckernel
