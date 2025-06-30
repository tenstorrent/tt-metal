// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

// Unary maximum
SFPU_UNARY_MAXMIN_KERNEL(unary_max, true, uint param0, param0)

// Unary minimum
SFPU_UNARY_MAXMIN_KERNEL(unary_min, false, uint param0, param0)

}  // namespace ckernel
