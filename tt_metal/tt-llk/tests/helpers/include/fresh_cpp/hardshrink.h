// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// hardshrink — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// hardshrink(x) = x where |x| > lambda, else 0 (PyTorch F.hardshrink reference
// semantics; golden_generators._hardshrink with the dispatch constant
// lambda = 0.5; |x| == lambda maps to 0 on both sides).  Production: metal
// calculate_hardshrink.  Fresh statement: the piecewise definition as one
// typed magnitude predicate.
#include <cstdint>

namespace ckernel::sfpu
{

// Fixed dispatch scalar, shared with the golden and identical to the value the
// production dispatch sends (sfpu_operations.h: 0x3f000000 /* lambda = 0.5f */;
// golden_generators._HARDSHRINK_LAMBDA).  Both legs must always receive the
// same value.
constexpr float FRESH_HARDSHRINK_LAMBDA = 0.5f;

template <int ITERATIONS>
__attribute__((noinline)) void calculate_hardshrink_fresh_cpp(const float lambda)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r       = v;
        v_if (sfpi::abs(v) <= lambda)
        {
            r = 0.0f;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
