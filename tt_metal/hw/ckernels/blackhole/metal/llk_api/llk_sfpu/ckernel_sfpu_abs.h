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
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // sfpi::abs(vInt) lowers to the dedicated SFPABS integer instruction (mod 0),
        // matching the raw TTI sequence (SFPLOAD + SFPABS + SFPSTORE = 3 SFPU ops).
        // On Blackhole INT32_2S_COMP load/store is a no-op vs INT32, so I32 access is
        // byte-for-byte equivalent to the previous mode-12 access. abs() yields a vMag,
        // which stores through the M32 layout.
        sfpi::vInt v = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::M32>() = sfpi::abs(v);
        sfpi::dst_reg++;
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
