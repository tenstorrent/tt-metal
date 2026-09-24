// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void alt_complex_rotate90_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_alt_complex_rotate90() {
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        dst_reg[0] = -vFloat(dst_reg[1]);
        dst_reg[1] = val;
        dst_reg += 2;
    }
}

// Op class for multiplying interleaved complex values by i. Same name and leading template parameters as
// on Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, int ITERATIONS = 4, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct AltComplexRotate90 : SfpuUnaryOp<AltComplexRotate90<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static inline __attribute__((always_inline)) void calculate() {
        calculate_alt_complex_rotate90<APPROXIMATION_MODE, ITERATIONS>();
    }
    static inline __attribute__((always_inline)) void init_op() { alt_complex_rotate90_init(); }
};

}  // namespace sfpu
}  // namespace ckernel
