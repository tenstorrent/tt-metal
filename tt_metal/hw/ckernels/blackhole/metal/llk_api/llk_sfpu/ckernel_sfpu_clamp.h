// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_unary_max_min.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

enum { Max = true, Min = false };  // Clamp Mode

inline void clamp_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

// out = min(max(x, min_val), max_val)
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp(uint min_val, uint max_val) {
    sfpi::vFloat lo = Converter::as_float(min_val);
    sfpi::vFloat hi = Converter::as_float(max_val);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        // A NaN input is left in place; clamp would return one of the bounds.
        v_if(sfpi::as<sfpi::vInt>(sfpi::setsgn(v, 0)) <= 0x7f800000) { sfpi::dst_reg[0] = sfpi::clamp(v, lo, hi); }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_clamp_int32(uint min_val, uint max_val) {
    for (int d = 0; d < ITERATIONS; d++) {
        load_value_param_int(min_val);
        calculate_unary_max_min_int32_body<Max>(min_val);
        load_value_param_int(max_val);
        calculate_unary_max_min_int32_body<Min>(max_val);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
