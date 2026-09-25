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
}  // namespace sfpu
}  // namespace ckernel

#if !defined(TT_POLY_LLK_DISABLE)
#include "ckernel_sfpu_abs_bf16.h"
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_abs_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_abs_value<ttpoly_generated::AbsBf16Config, ITERATIONS>();
}
inline void init_abs_tt_poly_bf16() { ckernel::sfpu::ttpoly::init_abs_value<ttpoly_generated::AbsBf16Config>(); }
#endif

}  // namespace ckernel::sfpu
