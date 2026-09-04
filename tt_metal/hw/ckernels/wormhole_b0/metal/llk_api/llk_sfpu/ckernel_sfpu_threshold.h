// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "sfpu/ckernel_sfpu_threshold.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// Threshold<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS, T>::calculate(dst_index, vector_mode, threshold, value)
//   backs threshold_tile / threshold_tile_init (bare per-op init). Thin metal wrapper over the tt-llk
//   _calculate_threshold_ kernel; T is the scalar parameter type (float or std::uint32_t).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8, typename T = std::uint32_t>
struct Threshold : SfpuUnaryOp<Threshold<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS, T>, DST_SYNC, DST_ACCUM> {
    static void kernel(T threshold, T value) {
        _calculate_threshold_<APPROXIMATION_MODE, ITERATIONS, T>(threshold, value);
    }
};

}  // namespace sfpu
}  // namespace ckernel
