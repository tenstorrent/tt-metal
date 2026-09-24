// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_rounding_ops.h"

namespace ckernel {
namespace sfpu {

// Op classes for the rounding kernels in tt-llk (sfpu/ckernel_sfpu_rounding_ops.h). They need no
// init beyond the op-agnostic SFPU init.

// Op class for elementwise floor.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Floor : SfpuUnaryOp<Floor<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = _calculate_floor_<APPROXIMATION_MODE, ITERATIONS>;
};

// Op class for elementwise ceil.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Ceil : SfpuUnaryOp<Ceil<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = _calculate_ceil_<APPROXIMATION_MODE, ITERATIONS>;
};

// Op class for elementwise truncation toward zero.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Trunc : SfpuUnaryOp<Trunc<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = _calculate_trunc_<APPROXIMATION_MODE, ITERATIONS>;
};

// Op class for the elementwise fractional part, x - trunc(x).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Frac : SfpuUnaryOp<Frac<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = _calculate_frac_<APPROXIMATION_MODE, ITERATIONS>;
};

// Op class for elementwise round-half-to-even to a given number of decimal places.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Round : SfpuUnaryOp<Round<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = _calculate_round_<APPROXIMATION_MODE, ITERATIONS>;
};

// Op class for elementwise stochastic rounding of FP32 to BF16.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct StochasticRound : SfpuUnaryOp<StochasticRound<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = _calculate_stochastic_round_<APPROXIMATION_MODE, ITERATIONS>;
};

}  // namespace sfpu
}  // namespace ckernel
