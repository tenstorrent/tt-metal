// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_cumsum.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_UNARY_PARAMS_KERNEL(cumsum, RC_custom, bool first, first)
