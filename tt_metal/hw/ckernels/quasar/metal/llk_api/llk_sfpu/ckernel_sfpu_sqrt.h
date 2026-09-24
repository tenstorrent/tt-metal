// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool EN_32BIT_DEST /*maybe_unused*/,
    bool FAST_APPROX /*maybe_unused*/ = false>
inline void calculate_sqrt() {
    static_assert(FAST_APPROX == false, "Non-default FAST_APPROX (true) not supported in Quasar sqrt");
    _calculate_sqrt_<APPROXIMATION_MODE, ITERATIONS>();
}

template <bool APPROXIMATION_MODE /*maybe_unused*/>
void sqrt_init() {
    // Empty function kept for backwards compatibility
}

// Op class for the square root.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    bool fp32_dest_acc_en = false,
    bool FAST_APPROX = false,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Sqrt : SfpuUnaryOp<Sqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX, SLOT>, SLOT> {
    static constexpr auto& calculate = calculate_sqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX>;
    static inline __attribute__((always_inline)) void init_op() { sqrt_init<APPROXIMATION_MODE>(); }
};

}  // namespace sfpu
}  // namespace ckernel
