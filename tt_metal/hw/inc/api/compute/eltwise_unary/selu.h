// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_selu.h"
#include "llk_math_eltwise_unary_sfpu_selu.h"
#endif

namespace ckernel {

ALWI void selu_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_selu<APPROX>(idst))); }

ALWI void selu_tile_init() { MATH((llk_math_eltwise_unary_sfpu_selu_init<APPROX>())); }

}  // namespace ckernel
