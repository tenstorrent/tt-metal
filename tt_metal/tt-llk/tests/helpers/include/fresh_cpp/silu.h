// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the silu op (storm contract, migrated
// verbatim from fresh_cpp_operations.h, Lane BR batch 1).
#include <cstdint>

namespace ckernel::sfpu
{

// Silu (production: _calculate_silu_ over the POLYVAL5 text macro with an
// abs/1-x symmetry fold; the row measures causal exactly 0.0% — the passes
// never engage the production structure).  Identical piecewise sigmoid math
// (the golden tolerance is fitted to it), restated with plain locals and a
// free loop so the compiler owns unrolling and delivery.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_silu_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v   = sfpi::dst_reg[0];
        const sfpi::vFloat mag = sfpi::abs(v);
        sfpi::vFloat sig       = 1.0f;
        v_if (mag <= 1.0f)
        {
            sig = mag * 0.229f + 0.5f;
        }
        v_elseif (mag < 5.0f)
        {
            sig = (((0.00144462f * mag + -0.01055479f) * mag + -0.01203685f) * mag + 0.24300185f) * mag + 0.50437757f;
        }
        v_endif;
        v_if (v < 0.0f)
        {
            sig = 1.0f - sig;
        }
        v_endif;
        sfpi::dst_reg[0] = v * sig;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
