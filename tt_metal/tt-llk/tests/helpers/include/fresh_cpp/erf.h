// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// erf — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// erf is odd and saturates to +/-1, so it is stated as x * P(x^2) with P a
// degree-7 least-squares fit of erf(x)/x on Chebyshev nodes over the sweep
// stimulus domain [0, 3] (fit derivation + tolerance validation:
// laneS2-evidence-20260819/fit_s2.py — max abs error 6.2e-4 against
// torch.erf, two orders under the suite's atol/rtol 0.05 gate), clamped to
// [-1, 1] so the tails stay monotone-saturated.  Golden: torch.erf
// (golden_generators._erf), Float32 corr contract.
//
// The fit is only valid on its own domain [-3, 3], so the ARGUMENT is reduced
// into that domain before P is evaluated: erf is monotone and saturating, and
// erf is within 2.3e-5 of its limit +/-1 for |x| >= 3, so erf(x) is stated as
// erf(clamp(x, -3, 3)) for the whole real line, and the |x| >= 3 tail is then
// stated as the LIMIT itself, exactly +/-1.  The limit is the better value
// there (|erf(3)| - 1 = 2.2e-5 against the fit's 6.4e-4 edge residue), it is
// what the [-1, 1] clamp below already meant by "monotone-saturated", and it
// keeps erfc's complement tails at exactly 0 and 2.  The argument reduction
// stays so the polynomial is never evaluated on an out-of-domain (or
// overflowing) value in the first place.  Measured max |err| vs torch.erf,
// fp32-faithful host model: 6.41e-4 on [-8, 8] and on [-40, 40], all of it
// inside |x| < 3 where the fit is the statement.  Without the
// argument reduction the degree-7 polynomial is evaluated far outside its
// domain, where the leading -5.5e-07 * x^14 term dominates and drives the
// product to the wrong sign: erf(11) came back as -1.0 instead of +1.0
// (x^2 = 121; and x^2 overflows to +inf for |x| >~ 1.8e19, giving -inf * x),
// 32513 bf16 ULP from the golden for every input in the stratum.
#include <cstdint>

namespace ckernel::sfpu
{

// Shared core: erf(x) = clamp(xc * P(xc^2), -1, 1) with xc = clamp(x, -3, 3)
// (the fit domain), saturated to the exact limit +/-1 outside it.
// erfc.h states its complement through this same core.
sfpi_inline sfpi::vFloat fresh_erf_core(const sfpi::vFloat x)
{
    // Saturating range reduction onto the fit domain (see the header note).
    constexpr float FIT_LIMIT = 3.0f;
    constexpr float E7        = -5.511776635e-07f;
    constexpr float E6        = 2.186222991e-05f;
    constexpr float E5        = -3.751075710e-04f;
    constexpr float E4        = 3.713592421e-03f;
    constexpr float E3        = -2.405889891e-02f;
    constexpr float E2        = 1.101540402e-01f;
    constexpr float E1        = -3.751232028e-01f;
    constexpr float E0        = 1.128316879e+00f;

    const sfpi::vFloat xc = sfpi::min(sfpi::max(x, -FIT_LIMIT), FIT_LIMIT);
    const sfpi::vFloat u  = xc * xc;
    sfpi::vFloat p        = E7;
    p                     = p * u + E6;
    p                     = p * u + E5;
    p                     = p * u + E4;
    p                     = p * u + E3;
    p                     = p * u + E2;
    p                     = p * u + E1;
    p                     = p * u + E0;
    sfpi::vFloat r        = xc * p;
    r                     = sfpi::min(r, 1.0f);
    r                     = sfpi::max(r, -1.0f);
    // Outside the fit domain erf IS its limit to within 2.3e-5, so state the
    // limit exactly instead of delivering the fit's edge value.
    v_if (sfpi::abs(x) >= FIT_LIMIT)
    {
        r = sfpi::copysgn(sfpi::vFloat(1.0f), x);
    }
    v_endif;
    return r;
}

template <int ITERATIONS>
__attribute__((noinline)) void calculate_erf_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::dst_reg[0] = fresh_erf_core(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
