// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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

inline void tiled_prod_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    vFloat result = 1.0f;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        result *= v;
        dst_reg[0] = result;
        dst_reg++;
    }
    vFloat v = dst_reg[0];
    result *= v;
    dst_reg[0] = result;
    dst_reg++;
}

// Op class for the running product over a tile. Same name and leading template parameters as on
// Wormhole/Blackhole.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct TiledProd : SfpuUnaryOp<TiledProd<APPROXIMATION_MODE, ITERATIONS, SLOT>, SLOT> {
    static constexpr auto& calculate = calculate_tiled_prod<APPROXIMATION_MODE, ITERATIONS>;
    static inline __attribute__((always_inline)) void init_op() { tiled_prod_init(); }
};

}  // namespace sfpu
}  // namespace ckernel
