// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "ckernel_sfpu_alt_complex_rotate90.h"

namespace ckernel {

// New LLK SFPU APIs

SFPU_INIT(alt_complex_rotate90)

SFPU_CALCULATE(alt_complex_rotate90)

}  // namespace ckernel
