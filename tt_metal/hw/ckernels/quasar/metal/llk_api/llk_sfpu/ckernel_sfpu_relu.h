// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_relu.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel {
namespace sfpu {

// ---------------------------------------------------------------------------------------------------
// Approach A dispatch structs (prototype). Same interface as WH/BH; the Quasar kernels take only an
// iteration count, so APPROXIMATION_MODE is accepted and ignored here and the arch difference stays
// out of the compute API header.
// ---------------------------------------------------------------------------------------------------
template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    DataFormat FORMAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = SFPU_ITERATIONS>
struct Relu : SfpuUnaryOp<Relu<APPROXIMATION_MODE, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static_assert(
        FORMAT == DataFormat::Float16_b || FORMAT == DataFormat::Float32, "Quasar relu supports float dest only");

    static void kernel() { _relu_min_<ITERATIONS>(0); }
};

template <
    [[maybe_unused]] bool APPROXIMATION_MODE,
    bool IS_LOWER_BOUND,
    DataFormat FORMAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = SFPU_ITERATIONS>
struct ReluClamp : SfpuUnaryOp<
                       ReluClamp<APPROXIMATION_MODE, IS_LOWER_BOUND, FORMAT, DST_SYNC, DST_ACCUM, ITERATIONS>,
                       DST_SYNC,
                       DST_ACCUM> {
    static_assert(
        FORMAT == DataFormat::Float16_b || FORMAT == DataFormat::Float32,
        "Quasar relu_min/relu_max support float dest only");

    static void kernel(uint32_t threshold) {
        if constexpr (IS_LOWER_BOUND) {
            _relu_min_<ITERATIONS>(threshold);
        } else {
            _relu_max_<ITERATIONS>(threshold);
        }
    }
};

}  // namespace sfpu
}  // namespace ckernel
