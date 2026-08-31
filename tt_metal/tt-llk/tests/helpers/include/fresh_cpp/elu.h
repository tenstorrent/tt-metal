// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// elu — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// elu(x) = x for x >= 0, alpha*(exp(x) - 1) for x < 0 with the dispatch
// constant alpha = 1 (PyTorch F.elu reference semantics; golden_generators
// ._elu).  The negative arm is expm1 stated by the shared Cody-Waite
// statement (fresh_common.h fresh_expm1_cw — the production elu/celu/selu
// family's expm1_cw_clamped numerics; range reduction = the
// tt-polynomial-fitter deployment canon, h(r) = the production Sollya fit).
//
// Lane JU coefficient repair (2026-08-31): the previous body computed the
// negative arm as exp_21f(x) - 1, whose exp approximation carries a +1.72e-3
// bias at 0 — the lane JN exhaustive certificate showed sem elu(±0) = +0x3ae2
// where production returns 0 (16,161/65,536 diverging inputs; also laneJL's
// "elu sem 7.9x less accurate than hand" finding).  expm1 stated directly is
// exact at the origin.  The arm is computed on all lanes and the x >= 0 lanes
// are overwritten (production shape); the alpha = 1 multiply folds away at
// compile time exactly as production's constant-propagated dispatch does.
// bf16 corr contract (Float16_b sweep row).
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

template <int ITERATIONS>
__attribute__((noinline)) void calculate_elu_fresh_cpp()
{
    for (int row = 0; row < ITERATIONS; ++row)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        // Negative arm on all lanes (vector select below): the positive
        // lanes' expm1 value is never observed.
        sfpi::vFloat r = fresh_expm1_cw(x);
        v_if (x >= 0.0f)
        {
            r = x;
        }
        v_endif;
        // bf16 destination: round to nearest-even before the store truncates.
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
