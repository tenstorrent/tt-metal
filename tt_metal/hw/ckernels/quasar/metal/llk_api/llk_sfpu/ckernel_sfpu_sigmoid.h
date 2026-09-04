// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_sigmoid.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_sigmoid() {
    _calculate_sigmoid_<ITERATIONS>();
}

// Sigmoid<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: sigmoid_tile / sigmoid_tile_init (compute_kernel_api.h).
// The Quasar kernel takes only an iteration count and uses the bare per-op init, so APPROXIMATION_MODE
// and DST_ACCUM are accepted for interface parity with WH/BH and ignored here.
template <[[maybe_unused]] bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = SFPU_ITERATIONS>
struct Sigmoid : SfpuUnaryOp<Sigmoid<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_sigmoid<ITERATIONS>(); }
};

}  // namespace sfpu
}  // namespace ckernel
