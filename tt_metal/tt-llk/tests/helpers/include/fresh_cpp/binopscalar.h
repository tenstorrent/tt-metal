// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `binopscalar` corpus row (metal
// calculate_binop_with_scalar).  The row's nodes drive the ScalarAdd arm:
// out = x + s, with the scalar arriving as raw fp32 bits exactly as the
// production dispatch sends it (the golden decodes the same pattern).
// The other arms (sub/mul/div/rsub) share the dispatch and keep their
// production-only coverage — one representative mathop per row.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_binop_scalar_add_fresh_cpp(const std::uint32_t value)
{
    const sfpi::vFloat scalar = Converter::as_float(value);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0]     = v + scalar;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
