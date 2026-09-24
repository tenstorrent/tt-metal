// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {
namespace sfpu {

// Element-wise negate for floats: out = -x. Identical to the Blackhole kernel: the plain sfpi store
// writes through the default (no-increment) address mode and dst_reg++ advances the Dest write
// counter by SFP_DESTREG_STRIDE (== SFP_ROWS == 2 on Quasar), so only the shared SFPU init is needed
// (no op-specific ADDR_MOD). There is no approximate variant (sign flip is exact), so
// APPROXIMATION_MODE is accepted for ABI parity but ignored.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

// Op class for float negation. Same name and leading template parameters as on Wormhole/Blackhole.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Negative : SfpuUnaryOp<Negative<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static inline __attribute__((always_inline)) void calculate() {
        _calculate_negative_<APPROXIMATION_MODE, ITERATIONS>();
    }
};

// Element-wise negate for int32: out = -x (two's-complement negate), for the negative_tile_int32 path.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_negative_int_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = -val;
        sfpi::dst_reg++;
    }
}

// Op class for int32 negation. Same name and leading template parameters as on Wormhole/Blackhole.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct NegativeInt : SfpuUnaryOp<NegativeInt<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static inline __attribute__((always_inline)) void calculate() {
        _calculate_negative_int_<APPROXIMATION_MODE, ITERATIONS>();
    }
};

}  // namespace sfpu
}  // namespace ckernel
