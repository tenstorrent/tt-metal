
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_log.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_INIT_LITERAL_KERNEL(log, sfpu::log_init, 0)
SFPU_UNARY_PARAMS_KERNEL(log_with_base, RC, uint base_scale, base_scale)
