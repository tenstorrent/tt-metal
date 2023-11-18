// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_template.h"
#include "metal_ckernel_sfpu.h"
#include "cmath_common.h"
#include "llk_format_conversions.h"
#include "llk_math_common.h"
#include "llk_param_structs.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;
