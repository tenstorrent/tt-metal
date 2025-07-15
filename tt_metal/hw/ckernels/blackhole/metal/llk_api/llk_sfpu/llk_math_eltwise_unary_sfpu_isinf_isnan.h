// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_isinf_isnan.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

// isinf
SFPU_ISINF_ISNAN_KERNEL(isinf, isinf)

// isposinf
SFPU_ISINF_ISNAN_KERNEL(isposinf, isposinf)

// isneginf
SFPU_ISINF_ISNAN_KERNEL(isneginf, isneginf)

// isnan
SFPU_ISINF_ISNAN_KERNEL(isnan, isnan)

// isfinite
SFPU_ISINF_ISNAN_KERNEL(isfinite, isfinite)
