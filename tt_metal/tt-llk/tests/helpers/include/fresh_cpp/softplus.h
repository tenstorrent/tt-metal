// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for softplus; see fresh_cpp/README.md for its contract.

#include <cstdint>

// The aggregator provides the shared fresh_exp_21f helper from helpers.h.
#include "fresh_cpp_operations.h"

namespace ckernel::sfpu
{

// softplus(t)/beta = (max(t, 0) + log1p(exp(-abs(t)))) / beta.
// The residual is the whole result for negative t, so it cannot be discarded
// as a small correction. The bf16 arm fits log1p(u)/u with the endpoint
// values 1 and log(2), where u is the zero-normalized exponential estimate.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_softplus_fresh_cpp(const float beta, const float beta_reciprocal, const float threshold)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat t         = beta * val;
        // Linear region softplus(t)/beta = val: explicit identity arm.
        sfpi::vFloat result = val;
        v_if (t < threshold)
        {
            const sfpi::vFloat a = sfpi::abs(t);
#ifdef INP_FLOAT32
            // f(a) = ln(1+exp(-a)) on [0, 5], degree 8 (production constants).
            sfpi::vFloat residual = 6.9310557842e-01f;
            {
                sfpi::vFloat p = -4.8245715334e-07f;
                p              = p * a + 2.1285692128e-05f;
                p              = p * a + -3.4358495031e-04f;
                p              = p * a + 2.7290175203e-03f;
                p              = p * a + -1.0528374463e-02f;
                p              = p * a + 5.6753782555e-03f;
                p              = p * a + 1.2186349183e-01f;
                p              = p * a + -4.9926245213e-01f;
                residual       = p * a + 6.9310557842e-01f;
            }
            // log1p(e) = e - e^2/2 + e^3/3 + O(e^4), e = exp(-a).
            v_if (a > 5.0f)
            {
                const sfpi::vFloat e = fresh_exp_21f(-a);
                residual             = e * (1.0f + e * (-0.5f + e * 0.333333343f));
            }
            v_endif;
#else
            // Degree-3 endpoint-constrained fit of log1p(u)/u on [0, 1].
            // This call divides the exp mantissa polynomial by its original
            // C0, giving e(0)=1; the residual can therefore use q directly.
            // Reproduce with corpus/tools/fit_softplus_residual.py.
            const sfpi::vFloat e        = fresh_exp_21f<true, true>(-a);
            sfpi::vFloat p              = -0.0752502828836441f;
            p                           = p * e + 0.25691255927085876f;
            p                           = p * e + -0.48851507902145386f;
            p                           = p * e + 1.0f;
            const sfpi::vFloat residual = e * p;
#endif
            const sfpi::vFloat sp = sfpi::max(t, 0.0f) + residual;
            result                = sfpi::convert<sfpi::vFloat16b>(beta_reciprocal * sp, sfpi::RoundMode::Nearest);
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
