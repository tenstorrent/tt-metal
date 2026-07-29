// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Fast inverse square-root seed + Newton-Raphson refinements.
// NEWTON_ITERATIONS controls the number of refinements: the bf16-magic seed
// (0x5f37) is ~3-4 correct bits, so each NR step roughly doubles the correct
// bits (seed -> ~7 bits after 1 iter -> ~14 bits after 2 iters). Two iterations
// give near-fp32 accuracy; a single iteration (~0.17% rel error, <0.5 bf16 ULP)
// suffices for consumers whose own approximation error already dominates.
template <bool APPROXIMATION_MODE, int NEWTON_ITERATIONS = 2>
sfpi_inline sfpi::vFloat sfpu_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;
    v_if(val != 0.0f) {
        sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
#pragma GCC unroll 2
        for (int i = 0; i < NEWTON_ITERATIONS; i++) {
            approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        }
        out = approx * val;
    }
    v_endif;
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
