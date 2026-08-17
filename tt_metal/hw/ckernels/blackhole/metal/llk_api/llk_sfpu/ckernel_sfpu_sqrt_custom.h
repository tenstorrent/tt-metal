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
    // Zero and the non-finite inputs all pass `val` straight through, which is already the
    // answer for +/-0 and +inf. Non-finite has to be excluded because the seed below is
    // ~5.2e-20 for +inf; squaring it underflows to a denormal, SFPMAD flushes that to +0,
    // and the next multiply is 0 * -inf = NaN -- so sqrt_custom(+inf) was NaN, and every
    // consumer inherited it (erfinv(+/-1)). Tested on the biased exponent field rather than
    // a compare against infinity because SFPSETCC is only specified for inputs that are not
    // NaN (VectorUnit.md), and this predicate has to be evaluated on a possible NaN.
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
