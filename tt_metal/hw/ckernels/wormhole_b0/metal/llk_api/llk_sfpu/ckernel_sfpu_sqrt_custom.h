// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

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
    // Skipped lanes pass `val` through, already the answer for +/-0 and +inf. Non-finite needs
    // excluding because the +inf seed (~5.2e-20) squares to a denormal, SFPMAD flushes it to +0,
    // and 0 * -inf = NaN: sqrt_custom(+inf) was NaN and consumers inherited it (erfinv(+/-1)).
    //
    // Exponent test rather than a compare against inf: SFPSETCC's float compare is unspecified
    // for NaN (VectorUnit.md), and `&&` is SFPXBOOL(AND), so `val != 0.0f` is still evaluated on
    // NaN lanes. That is safe -- exexp(NaN) == 255 falsifies the other conjunct and AND is
    // monotone, so NaN passes through whatever the compare returned.
    //
    // -inf is the one non-finite the pass-through gets wrong: IEEE sqrt(-inf) = NaN. Guarding
    // on the exact -inf bit pattern rather than on `val < 0` keeps erfinv safe -- its NR
    // undershoot makes `tmp + intermediate_result` (ckernel_sfpu_erfinv.h:40) slightly negative
    // for small in-domain x, and a negative-to-NaN guard would turn erfinv(1e-6) into NaN.
    //
    // The NaN has to be written as a literal +qNaN, not synthesised as (-inf + inf): that
    // arithmetic yields a *negative* NaN here, and the pack path substitutes an inf of the same
    // sign, so the result lands as -inf on every bf16 Dest or bf16 output.
    v_if(val != 0.0f && sfpi::exexp(val, sfpi::ExponentMode::Biased) != 255) {
        sfpi::vUInt magic = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::sFloat16b(0x5f37)));
        sfpi::vFloat approx = sfpi::as<sfpi::vFloat>(magic - (sfpi::as<sfpi::vUInt>(val) >> 1));
        sfpi::vFloat neg_half_val = val * -0.5f;
#pragma GCC unroll 2
        for (int i = 0; i < NEWTON_ITERATIONS; i++) {
            approx = ((approx * approx) * neg_half_val + 1.5f) * approx;
        }
        out = approx * val;
    }
    v_elseif(sfpi::as<sfpi::vInt>(val) == static_cast<int>(0xFF800000)) {
        out = std::numeric_limits<float>::quiet_NaN();
    }
    v_endif;
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
