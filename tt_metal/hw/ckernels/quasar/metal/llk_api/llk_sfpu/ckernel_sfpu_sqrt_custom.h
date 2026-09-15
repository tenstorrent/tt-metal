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
    // Exclude non-finite: the +inf seed squares to a denormal, SFPMAD flushes it to +0, and
    // 0 * -inf = NaN. Same guard as Wormhole and Blackhole, see #53757.
    v_if(val != 0.0f && sfpi::exexp(val, sfpi::ExponentMode::Biased) != 255) {
        // Fast inverse square-root seed + two Newton-Raphson refinements.
        sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        out = approx * val;
    }
    v_endif;
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
