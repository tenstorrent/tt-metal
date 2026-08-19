// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the threshold op (storm contract:
// fresh_cpp/README.md).  Independent derivation from the PyTorch reference
// (torch.nn.functional.threshold — the production golden):
//
//   threshold(x) = x      if x >  t
//                  value  otherwise
//
// stated with the same predicate direction as the kernel contract
// (x <= t -> value; ties replace).  The production hand-isms removed: the
// "#pragma GCC unroll 8" pin and the store-only-under-predicate pattern —
// here one value, one unconditional store, a free loop.

#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_threshold_fresh_cpp(const float threshold, const float value)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if (v <= threshold)
        {
            v = value;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
