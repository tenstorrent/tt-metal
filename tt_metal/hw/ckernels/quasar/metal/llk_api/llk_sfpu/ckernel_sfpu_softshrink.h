// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel_sfpu_converter.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_compat.h"

namespace ckernel::sfpu {

inline void softshrink_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(std::uint32_t param0) {
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // SFPU microcode
    sfpi::vFloat lambda = Converter::as_float(param0);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = 0.0f;
        v_if(v > lambda) { sfpi::dst_reg[0] = v - lambda; }
        v_elseif(compat::fp_lt(v, -lambda)) { sfpi::dst_reg[0] = v + lambda; }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
