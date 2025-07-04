// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_gelu.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

// the "forward" GELU
SFPU_INIT_KERNEL(gelu, sfpu::gelu_init)

// the GELU derivative
SFPU_INIT_KERNEL(gelu_derivative, sfpu::gelu_derivative_init)
