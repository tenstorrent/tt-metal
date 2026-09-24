// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softsign() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat tmp = sfpi::abs(v) + 1.0f;
        tmp = _sfpu_reciprocal_<(APPROXIMATION_MODE) ? 0 : 2>(tmp);
        sfpi::dst_reg[0] = v * tmp;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void init_softsign() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    _init_reciprocal_<APPROXIMATION_MODE>();
}

// Op class for softsign: x / (1 + |x|).
// Same name and leading template parameters as on Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Softsign : SfpuUnaryOp<Softsign<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static inline __attribute__((always_inline)) void calculate() {
        calculate_softsign<APPROXIMATION_MODE, ITERATIONS>();
    }
    static inline __attribute__((always_inline)) void init_op() { init_softsign<APPROXIMATION_MODE>(); }
};

}  // namespace ckernel::sfpu
