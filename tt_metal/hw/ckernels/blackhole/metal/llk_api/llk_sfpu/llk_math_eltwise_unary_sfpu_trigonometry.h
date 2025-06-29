// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_trigonometry.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_TRIG_KERNEL(sine)
SFPU_TRIG_KERNEL(cosine)
SFPU_TRIG_KERNEL(tan)

SFPU_UNARY_KERNEL(asin)
SFPU_UNARY_KERNEL(acos)

SFPU_INVERSE_HYPERBOLIC_KERNEL(acosh, 8)

SFPU_INIT_KERNEL(atan, sfpu::atan_init)
SFPU_UNARY_KERNEL(atan)

SFPU_INVERSE_HYPERBOLIC_KERNEL(asinh, 8)
