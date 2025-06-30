// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_hardtanh.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_UNARY_PARAMS_KERNEL_NO_INITCB(hardtanh, RC, uint param0, uint param1, uint param2, param0, param1, param2)
