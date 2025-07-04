// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_comp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

// EQZ
SFPU_ZERO_KERNEL(eqz, equal_zero, 8)

// NEZ
SFPU_ZERO_KERNEL(nez, not_equal_zero, 8)

// LTZ
SFPU_ZERO_KERNEL(ltz, less_than_zero, 8)

// GTZ
SFPU_ZERO_KERNEL(gtz, greater_than_zero, 8)

// LEZ
SFPU_ZERO_KERNEL(lez, less_than_equal_zero, 8)

// GEZ
SFPU_ZERO_KERNEL(gez, greater_than_equal_zero, 8)

}  // namespace ckernel
