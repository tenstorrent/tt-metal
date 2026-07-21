// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

inline void hardtanh_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// Hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x
// Equivalent to: clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardtanh(uint param0, uint param1) {
    // Materialize both bounds outside the loop for better performance
    sfpi::vFloat min_val = Converter::as_float(param0);
    sfpi::vFloat max_val = Converter::as_float(param1);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::clamp(v, min_val, max_val);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
