// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat sfpu_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;
    v_if(val != 0.0f) {
        // Fast inverse square-root seed + two Newton-Raphson refinements.
        sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
        // Newton-Raphson y <- y*(1.5 - 0.5*val*y^2). The residual is evaluated as
        // (approx * neg_half_val) * approx and NOT (approx * approx) * neg_half_val:
        // approx^2 = 1/val underflows fp32 to zero once val exceeds ~2**64 (the SFPU
        // flushes subnormals), which silently reduced each iteration to a bare *1.5
        // and made sqrt(3.3e38) 2.16x too large. The reassociated product keeps both
        // factors normal over the whole fp32 range and is otherwise identical.
        approx = ((approx * neg_half_val) * approx + 1.5f) * approx;
        approx = ((approx * neg_half_val) * approx + 1.5f) * approx;
        out = approx * val;
    }
    v_endif;
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
