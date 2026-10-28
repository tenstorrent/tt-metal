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

/**
 * @brief Running product along each SFPU lane through faces 0-3 of one Dest tile, each partial
 * product stored in place.
 *
 * @tparam APPROXIMATION_MODE: Unused; kept for the SFPU dispatch signature.
 * @tparam ITERATIONS: Steps per face (two Dest rows each); must span exactly one face, enforced
 * by the static_assert below.
 * @note Run once per tile under VectorMode::RC_custom, not once per face. The product is carried
 * in a register from face to face, so every store lands inside the tile being processed; a
 * per-face call with a ninth, carry-over store would write the first rows of the next Dest tile,
 * which belongs to another tile of the batch or to the packer's half.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tiled_prod() {
    // One call must cover exactly one tile: each step advances dst_reg by SFP_DESTREG_STRIDE Dest
    // rows, so ITERATIONS steps have to span one face. Fewer leave part of the tile untouched,
    // more walk into the next Dest tile.
    static_assert(
        ITERATIONS * static_cast<int>(SFP_DESTREG_STRIDE) == static_cast<int>(FACE_R_DIM),
        "calculate_tiled_prod: ITERATIONS steps of SFP_DESTREG_STRIDE Dest rows must cover one face");
    vFloat result = 1.0f;
#pragma GCC unroll 8
    for (int d = 0; d < static_cast<int>(TILE_NUM_FACES) * ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        result *= v;
        dst_reg[0] = result;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
