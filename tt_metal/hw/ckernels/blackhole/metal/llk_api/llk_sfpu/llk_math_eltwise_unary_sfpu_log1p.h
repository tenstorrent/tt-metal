// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_log1p.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_INIT_KERNEL(log1p, sfpu::log1p_init)
