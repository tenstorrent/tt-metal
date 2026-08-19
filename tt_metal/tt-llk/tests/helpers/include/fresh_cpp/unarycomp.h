// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the unarycomp op (storm contract:
// fresh_cpp/README.md).  Independent derivation from the mathematical
// definition (the production golden: 1.0 where the comparison holds, else
// 0.0, against a fixed scalar):
//
//   unary_ge(x) = (x >= s) ? 1 : 0
//   unary_le(x) = (x <= s) ? 1 : 0
//
// One shared body, the comparison direction a template parameter (the
// unary max/min precedent).  The production hand-ism removed: the
// "#pragma GCC unroll 8" pin.  NOTE on NaN: the production kernels spell
// ge as !(x < s), which sends NaN lanes to 1.0; this body predicates on
// the definition itself, which sends NaN to 0.0 — the golden's value.
// The swept domain contains no NaN, so the two agree everywhere tested.

#include <cstdint>

namespace ckernel::sfpu
{

template <bool IS_GE, int ITERATIONS>
__attribute__((noinline)) void calculate_unary_comp_fresh_cpp(const std::uint32_t value)
{
    const float s = Converter::as_float(value);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat r       = 0.0f;
        if constexpr (IS_GE)
        {
            v_if (x >= s)
            {
                r = 1.0f;
            }
            v_endif;
        }
        else
        {
            v_if (x <= s)
            {
                r = 1.0f;
            }
            v_endif;
        }
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
