// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_sqrt.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = SFPU_ITERATIONS,
    [[maybe_unused]] bool EN_32BIT_DEST,
    [[maybe_unused]] bool FAST_APPROX = false>
inline void calculate_sqrt() {
    static_assert(FAST_APPROX == false, "Non-default FAST_APPROX (true) not supported in Quasar sqrt");
    _calculate_sqrt_<APPROXIMATION_MODE, ITERATIONS>();
}

template <[[maybe_unused]] bool APPROXIMATION_MODE>
void sqrt_init() {
    // Empty function kept for backwards compatibility
}

// ---------------------------------------------------------------------------------------------------
// Sqrt<APPROX, FAST_APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode) -> calculate_sqrt
//   init()                            -> sqrt_init
// Backs sqrt_tile / sqrt_tile_init.
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, bool FAST_APPROX, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Sqrt : SfpuUnaryOp<Sqrt<APPROXIMATION_MODE, FAST_APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel() { calculate_sqrt<APPROXIMATION_MODE, ITERATIONS, DST_ACCUM, FAST_APPROX>(); }

    static void init_kernel() { sqrt_init<APPROXIMATION_MODE>(); }
};
}  // namespace sfpu
}  // namespace ckernel
