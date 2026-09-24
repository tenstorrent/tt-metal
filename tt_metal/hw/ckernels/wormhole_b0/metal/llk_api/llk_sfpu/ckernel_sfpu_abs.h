// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void abs_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        dst_reg[0] = sfpi::abs(v);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    // Kept as raw TTI intrinsics on purpose: SFPABS computes int32 abs in a single SFPU op. The sfpi
    // `v & 0x7FFFFFFF` form (#48598) needed two extra per-element SFPLOADI to rebuild the mask inside
    // the replay block, costing +30% cyc/tile.
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, InstrModLoadStore::INT32, 3, 0);
        TTI_SFPABS(0, 1, 0, 0);
        TTI_SFPSTORE(0, InstrModLoadStore::INT32, 3, 0);
        dst_reg++;
    }
}

// Op class for |x| on float tiles.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Abs : SfpuUnaryOp<Abs<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = calculate_abs<APPROXIMATION_MODE, ITERATIONS>;
    static inline __attribute__((always_inline)) void init_op() { abs_init(); }
};

// Op class for |x| on int32 tiles.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct AbsInt32 : SfpuUnaryOp<AbsInt32<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = calculate_abs_int32<APPROXIMATION_MODE, ITERATIONS>;
    static inline __attribute__((always_inline)) void init_op() { abs_init(); }
};

}  // namespace sfpu
}  // namespace ckernel
