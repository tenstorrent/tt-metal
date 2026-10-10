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
// (x <= t -> value; ties replace).  The production unroll pin remains absent;
// the explicit NaN-safe predicate leaves complement lanes bit-untouched.

#include <cstdint>

namespace ckernel::sfpu
{

// NOTE on NaN: PyTorch propagates it.  SFPU comparisons total-order NaNs, so
// the finite-domain `v <= t` spelling needs an explicit final pass-through.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_threshold_fresh_cpp(const float threshold, const float value)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat input = sfpi::dst_reg[0];
        // Leave complement lanes untouched, including either-sign NaNs.
        v_if (!sfpi::is_nan(input) && input <= threshold)
        {
            sfpi::dst_reg[0] = value;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
