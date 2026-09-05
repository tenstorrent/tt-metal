// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_is_fp16_zero.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void sign_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sign(const uint /*exponent_size_8*/) {
// All params are in FP16 format
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat res = 1.0f;
        // SFPSETCC is unspecified for -0.0 (VectorUnit.md specifies the compare only for
        // inputs that are neither negative zero nor NaN), so (v < 0.0f) can capture -0.0
        // and _sfpu_is_fp16_zero_ can miss it. SFPABS clears the sign bit, which brings
        // the compare back inside the contract.
        v_if(sfpi::abs(v) == 0.0F) { res = 0.0f; }
        v_elseif(v < 0.0F) { res = -1.0f; }
        v_endif;
        sfpi::dst_reg[0] = res;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
