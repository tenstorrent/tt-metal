// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_silu.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_silu() {
    _calculate_silu_<ITERATIONS>();
}

// Silu<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>: silu_tile / silu_tile_init (compute_kernel_api.h).
// The Quasar kernel takes only an iteration count and uses the bare per-op init, so APPROXIMATION_MODE
// and DST_ACCUM are accepted for interface parity with WH/BH and ignored here.
template <[[maybe_unused]] bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = SFPU_ITERATIONS>
struct Silu : SfpuUnaryOp<Silu<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_silu<ITERATIONS>(); }
};

}  // namespace sfpu
}  // namespace ckernel
