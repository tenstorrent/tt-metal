// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
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
#include "sfpu/ckernel_sfpu_reshuffle_rows.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "sfpu/ckernel_sfpu_sign.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_square.h"
#include "sfpu/ckernel_sfpu_tanh_derivative.h"
#include "sfpu/ckernel_sfpu_tanh.h"
#include "sfpu/ckernel_sfpu_topk.h"
#include "sfpu/ckernel_sfpu_trigonometry.h"
#include "sfpu/ckernel_sfpu_typecast.h"

// using namespace sfpi;

// namespace ckernel
// {
// namespace sfpu
// {

/*
template <bool APPROXIMATION_MODE, bool ZERO_NEGATIVE, bool SCALE_EN>
void calculate_cube(uint16_t exp_base_scale_factor = 0)
{
    for (int d = 0; d < ITERATIONS; d++)
    {

        TTI_SFPLOAD(p_sfpu::LREG3, 0, 0); // load from dest
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPNOP; TTI_SFPNOP;
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG3, p_sfpu::LCONST_1, p_sfpu::LREG2, 0);
        TTI_SFPNOP; TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG2, 0, 0); // Store from lreg[1] into dest registers
        TTI_INCRWC(0, 2, 0, 0);
    }
}
*/

// } // namespace sfpu
// } // namespace ckernel
