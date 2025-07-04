// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_trigonometry.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// New LLK SFPU APIs

// sine
SFPU_TRIG_KERNEL(sine)

// cosine
SFPU_TRIG_KERNEL(cosine)

// tangent
SFPU_TRIG_KERNEL(tan)

// asin
SFPU_UNARY_KERNEL(asin)

// acos
SFPU_UNARY_KERNEL(acos)

// acosh
SFPU_INVERSE_HYPERBOLIC_KERNEL(acosh, 8)

// atan
SFPU_INIT_KERNEL(atan, sfpu::atan_init)

// asinh
SFPU_INVERSE_HYPERBOLIC_KERNEL(asinh, 8)

}  // namespace ckernel
