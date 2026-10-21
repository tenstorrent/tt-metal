// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat abs_v = sfpi::abs(v);
        sfpi::vFloat result;
        v_if(abs_v == std::numeric_limits<float>::infinity()) {
            result = sfpi::setsgn(1.0f, v);
        }
        v_else {
            sfpi::vFloat tmp = sfpu_reciprocal<APPROXIMATION_MODE>(abs_v + 1.0f);
            result = v * tmp;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
