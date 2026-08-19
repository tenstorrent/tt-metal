// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the softshrink op (storm contract:
// fresh_cpp/README.md).  Independent derivation from the PyTorch reference
// (torch.nn.functional.softshrink — the production golden):
//
//   softshrink(x) = x - lambda  if x >  lambda
//                   x + lambda  if x < -lambda
//                   0           otherwise      (strict inequalities: |x| == lambda -> 0)
//
// The production kernel (metal calculate_softshrink) stores dst_reg[0] up to
// three times per datum (a zero default rewritten under each predicate) —
// the hand-ism this body removes with one value and one unconditional store.

#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_softshrink_fresh_cpp(const std::uint32_t param0)
{
    const float lambda = Converter::as_float(param0);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat r       = 0.0f;
        v_if (x > lambda)
        {
            r = x - lambda;
        }
        v_elseif (x < -lambda)
        {
            r = x + lambda;
        }
        v_endif;
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
