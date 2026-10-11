// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the relu op file's ReluMax vehicle (storm
// contract, fresh_cpp/README.md).  Production: legacy ckernel_sfpu_relu.h
// _relu_max_impl_ — an un-unrolled runtime-count loop behind a VectorType/T
// dispatch wrapper.  Semantic statement of the golden
// (torch.relu(torch.min(x, threshold))): clamp above at the threshold, zero
// below 0. SFPU comparisons total-order NaNs, so a sign-cleared magnitude
// guard keeps NaNs on the input-initialized path and applies the min/max clamp
// only to non-NaNs. Finite outputs are the input value, the threshold, or 0,
// all exactly representable. NaNs stay on the input path; the existing BF16
// destination packer then canonicalizes them to signed infinities.
#include <cstdint>
#include <limits>

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
        // Clear the sign explicitly: SFPABS float mode deliberately preserves
        // the sign of negative NaNs. After SFPSETSGN, every NaN encoding is
        // bitwise above +inf while finite values and infinity are not. Keep
        // NaNs on the input-initialized path; SFPSWAP min/max total-order them.
        const sfpi::vFloat magnitude = sfpi::setsgn(input, 0);
        v_if (magnitude <= std::numeric_limits<float>::infinity())
        {
            v = sfpi::max(sfpi::min(input, threshold), 0.0f);
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
