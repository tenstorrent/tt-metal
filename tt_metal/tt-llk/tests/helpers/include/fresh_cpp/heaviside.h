// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// heaviside — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// H(x) = 0 for x < 0, 1 for x > 0, and the dispatch-supplied value at exactly
// x == 0 (PyTorch torch.heaviside reference semantics; golden_generators
// ._heaviside with the dispatch constant 0.5).  Production: metal
// calculate_heaviside.  Fresh statement: the piecewise definition as typed
// predicate regions.
#include <cstdint>

namespace ckernel::sfpu
{

// Fixed dispatch scalar, shared with the golden and identical to the value the
// production dispatch sends (sfpu_operations.h: 0x3f000000 /* value = 0.5f */;
// golden_generators._heaviside).  Both legs must always receive the same value.
constexpr float FRESH_HEAVISIDE_VALUE = 0.5f;

template <int ITERATIONS>
__attribute__((noinline)) void calculate_heaviside_fresh_cpp(const float value)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat r       = 0.0f;
        v_if (v > 0.0f)
        {
            r = 1.0f;
        }
        v_elseif (v == 0.0f)
        {
            r = value;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
