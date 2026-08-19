// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the softplus op (storm contract:
// fresh_cpp/README.md).  Migrated verbatim from fresh_cpp_operations.h
// (Lane BR batch 3); byte-stable algorithm, only the file moved.

#include <cstdint>

// Shared helper (fresh_exp_21f) still lives in the legacy header pending
// full migration (fresh_cpp/README.md legacy note).
#include "fresh_cpp_operations.h"

namespace ckernel::sfpu
{

// Softplus (production: calculate_softplus — bit-punned scalar params and a
// store-only-under-predicate linear region; the arithmetic itself is already
// constexpr-clean).  Same residual polynomials and tail handling per format
// arm (the format arm is the contract, selected the same way production
// selects it); the linear region is stated as an explicit identity arm with
// one unconditional store.  Scalars are the golden's fixed dispatch values.
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
            // Tail: f(a) ~= e - e^2/2 + e^3/3 with e = exp(-a).
            v_if (a > 5.0f)
            {
                const sfpi::vFloat e = fresh_exp_21f(-a);
                residual             = e * (1.0f + e * (-0.5f + e * 0.333333343f));
            }
            v_endif;
#else
            // Degree-6 bf16 fit (production constants); past the fit domain the
            // true residual is below bf16 rounding, so it clamps to zero.
            sfpi::vFloat residual;
            {
                sfpi::vFloat p = -3.1273466851e-05f;
                p              = p * a + 5.0152968088e-04f;
                p              = p * a + -1.8627923291e-03f;
                p              = p * a + -1.3000584069e-02f;
                p              = p * a + 1.4279095486e-01f;
                p              = p * a + -5.0932420424e-01f;
                residual       = p * a + 6.9423984729e-01f;
            }
            v_if (a > 5.0f)
            {
                residual = 0.0f;
            }
            v_endif;
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
