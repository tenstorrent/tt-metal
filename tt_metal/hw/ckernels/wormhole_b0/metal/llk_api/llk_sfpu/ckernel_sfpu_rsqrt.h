// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sqrt.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool fp32_dest_acc_en, int ITERATIONS = 8, bool FAST_APPROX = false>
inline void calculate_rsqrt() {
    _calculate_sqrt_internal_<!fp32_dest_acc_en, ITERATIONS, fp32_dest_acc_en, true, FAST_APPROX>();
}

template <bool fp32_dest_acc_en>
void rsqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sqrt_init<fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
