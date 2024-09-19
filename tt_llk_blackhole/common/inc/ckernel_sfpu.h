// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include <limits>
#include "ckernel_globals.h"

#include "sfpi.h"

#include "sfpu/ckernel_sfpu_abs.h"
#include "sfpu/ckernel_sfpu_add_int32.h"
#include "sfpu/ckernel_sfpu_cast_fp32_to_fp16a.h"
#include "sfpu/ckernel_sfpu_clamp.h"
#include "sfpu/ckernel_sfpu_comp.h"
#include "sfpu/ckernel_sfpu_cumsum.h"
#include "sfpu/ckernel_sfpu_dropout.h"
#include "sfpu/ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_gelu.h"
#include "sfpu/ckernel_sfpu_hardtanh.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_max_int32.h"
#include "sfpu/ckernel_sfpu_max.h"
#include "sfpu/ckernel_sfpu_power.h"
#include "sfpu/ckernel_sfpu_quant.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_sign.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_square.h"
#include "sfpu/ckernel_sfpu_tanh_derivative.h"
#include "sfpu/ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "sfpu/ckernel_sfpu_trigonometry.h"
#include "sfpu/ckernel_sfpu_typecast.h"
