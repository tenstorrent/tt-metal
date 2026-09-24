// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_silu() {
    _calculate_silu_<ITERATIONS>();
}

// Op class for silu(x). Same name and leading template parameters as on Wormhole/Blackhole; the Quasar
// kernel does not depend on APPROXIMATION_MODE or is_fp32_dest_acc_en.
template <
    bool APPROXIMATION_MODE,
    bool is_fp32_dest_acc_en = false,
    int ITERATIONS = SFPU_ITERATIONS,
    trisc::DstTileShape SLOT = trisc::DstTileShape::Tile32x32>
struct Silu : SfpuUnaryOp<Silu<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS, SLOT>, SLOT> {
    static inline __attribute__((always_inline)) void calculate() { calculate_silu<ITERATIONS>(); }
};

}  // namespace sfpu
}  // namespace ckernel
