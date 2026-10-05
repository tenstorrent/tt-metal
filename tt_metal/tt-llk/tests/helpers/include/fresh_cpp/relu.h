// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the relu op file's ReluMax vehicle (storm
// contract, fresh_cpp/README.md).  Production: legacy ckernel_sfpu_relu.h
// _relu_max_impl_ — an un-unrolled runtime-count loop behind a VectorType/T
// dispatch wrapper.  Semantic statement of the golden
// (torch.relu(torch.min(x, threshold))): clamp above at the threshold, zero
// below 0.  SFPU comparisons total-order NaNs, so an explicit final predicate
// restores the mathematical NaN pass-through.  Outputs are the input value,
// the threshold, or 0 — all exactly representable — so no store rounding.
#include <cstdint>

namespace ckernel::sfpu
{

// Fixed dispatch threshold shared with the golden and the production
// dispatch (sfpu_operations.h 5.0f == sfpu_dispatch_constants.py
// RELU_MAX_THRESHOLD).  Both legs must always receive identical values.
constexpr float FRESH_RELU_MAX_THRESHOLD = 5.0f;

template <int ITERATIONS>
__attribute__((noinline)) void calculate_relu_max_fresh_cpp(const float threshold)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat input = sfpi::dst_reg[0];
        sfpi::vFloat v           = input;
        v_if (v > threshold)
        {
            v = threshold;
        }
        v_endif;
        v_if (v < 0.0f)
        {
            v = 0.0f;
        }
        v_endif;
        // The hardware comparisons total-order NaNs and would otherwise
        // clamp them.  ReluMax propagates NaN under its mathematical contract.
        v_if (sfpi::is_nan(input))
        {
            v = input;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
