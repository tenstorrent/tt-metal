// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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
//
// NEGATIVE_INFINITY_SAFE adds a branch for IEEE sqrt(-inf) = NaN; off by default, see below.
// It is appended rather than inserted so that every pre-existing spelling keeps its meaning --
// `<false>`, `<false, 1>` and `<false, 2>` all still bind the iteration count. Blackhole carries
// the same three parameters in the same order, so the shared test wrapper needs no #ifdef.
template <bool APPROXIMATION_MODE, int NEWTON_ITERATIONS = 2, bool NEGATIVE_INFINITY_SAFE = false>
sfpi_inline sfpi::vFloat sfpu_sqrt_custom(sfpi::vFloat in) {
    sfpi::vFloat val = in;
    sfpi::vFloat out = val;
    // Skipped lanes pass `val` through, already the answer for +/-0 and +inf. Non-finite needs
    // excluding because the +inf seed (~5.2e-20) squares to a denormal, SFPMAD flushes it to +0,
    // and 0 * -inf = NaN: sqrt_custom(+inf) was NaN and consumers inherited it (erfinv(+/-1)).
    //
    // Exponent test rather than a compare against inf: SFPSETCC's float compare is unspecified
    // for NaN (VectorUnit.md), and `&&` is SFPXBOOL(AND), so `val != 0.0f` is still evaluated on
    // NaN lanes. That is safe -- is_finite(NaN) is false and AND is monotone, so NaN passes
    // through whatever the compare returned.
    v_if(val != 0.0f && sfpi::is_finite(val)) {
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

    // IEEE sqrt(-inf) = NaN, where the pass-through above yields -inf. Opt-in, because the
    // branch is not free and no production consumer can reach a -inf: erfinv's arguments are
    // bounded (>= 18.75 and > -4.33) and asin/acos only pass (1 - |v|) * 0.5 from inside
    // v_if(abs(val) <= 1.0f). Measured on a WH n150, mean(MATH_ISOLATE), ELF cache wiped
    // between arms: always-on costs erfinv 1.127x and asin 1.067x. Folding it into the v_if
    // above as a v_elseif arm measures 1.128x and an exexp/sign pair 1.149x, so the separate
    // exact-pattern test is the cheapest shape there is -- it is the predicate region that
    // costs, not the compare. The default therefore still returns -inf, and only the test-only
    // calculate_sqrt_custom wrapper opts in; sfpu_domains.py records that split.
    //
    // Must stay an exact bit-pattern test writing a literal +qNaN. `val < 0.0f` would claim
    // -0.0 as well -- SFPSETCC reads the sign bit, and sqrt_custom(-0) must stay -0 -- and a NaN
    // synthesised with val's sign would be a negative NaN, which a bf16 pack turns into -inf.
    if constexpr (NEGATIVE_INFINITY_SAFE) {
        constexpr std::int32_t negative_infinity_bits = static_cast<std::int32_t>(0xFF800000);
        v_if(sfpi::as<sfpi::vInt>(val) == negative_infinity_bits) { out = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    return out;
}

}  // namespace sfpu
}  // namespace ckernel
