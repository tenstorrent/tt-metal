// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_UNARY_KERNEL_INIT_UNUSED(rounding_op)

SFPU_ROUNDING_OP_KERNEL(floor)

SFPU_ROUNDING_OP_KERNEL_FLOAT32(floor)

SFPU_ROUNDING_OP_KERNEL(ceil)

SFPU_ROUNDING_OP_KERNEL_FLOAT32(ceil)

SFPU_TRUNC_OP_KERNEL(trunc)

SFPU_TRUNC_OP_KERNEL_FLOAT32(trunc)

SFPU_ROUND_WITH_DECIMALS_KERNEL(round)

}  // namespace ckernel
