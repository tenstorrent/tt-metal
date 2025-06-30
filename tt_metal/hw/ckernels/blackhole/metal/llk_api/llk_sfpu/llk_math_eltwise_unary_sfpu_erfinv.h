// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_erfinv.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

SFPU_INIT_KERNEL(erfinv, sfpu::erfinv_init)
SFPU_OP_SUFFIX_KERNEL(erfinv)
