// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void tiled_prod_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

// Runs once per tile (VectorMode::RC_custom): one running product per SFPU lane over all four
// faces, ITERATIONS steps (two Dest rows each) per face. The product is carried in a register
// from face to face, so every store lands inside the tile being processed; a per-face call with
// a ninth, carry-over store would write the first rows of the next Dest tile, which belongs to
// another tile of the batch or to the packer's half.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    vFloat result = 1.0f;
#pragma GCC unroll 8
    for (int d = 0; d < 4 * ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        result *= v;
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
