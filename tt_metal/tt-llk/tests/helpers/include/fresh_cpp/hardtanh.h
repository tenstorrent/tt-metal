// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// hardtanh — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// Migrated verbatim from ../fresh_cpp_operations.h (Lane BR causal-tier lift);
// self-contained (no shared-helper dependency).
#include <cstdint>

namespace ckernel::sfpu
{

// Hardtanh (production: _calculate_hardtanh_ encodes the clamp as three
// chained add-then-zero-select steps over host-pre-negated bf16 params).
// hardtanh(x) = clamp(x, lo, hi) — same golden, same bounds; stated directly.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_hardtanh_fresh_cpp(const float lo, const float hi)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v = sfpi::dst_reg[0];
        v_if (v < lo)
        {
            v = lo;
        }
        v_elseif (v >= hi)
        {
            v = hi;
        }
        v_endif;
        sfpi::dst_reg[0] = v;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
