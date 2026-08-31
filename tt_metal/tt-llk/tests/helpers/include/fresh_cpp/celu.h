// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `celu` corpus row (metal
// calculate_celu).  Mathematical definition (torch.nn.functional.celu):
//
//   celu(x, alpha) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
//                  = x                          for x >= 0
//                  = alpha * (exp(x/alpha) - 1) for x <  0
//
// alpha and 1/alpha are the dispatch scalars exactly as the production
// dispatch sends them (both 1.0f; the golden uses alpha = 1.0).  Production
// receives them as literals into an inline kernel, so they constant-fold and
// the x/alpha and alpha* multiplies vanish at compile time; this body takes
// them as template constants so the same folding happens here (a runtime
// 1.0f multiply is NOT value-neutral on this pipeline: SFPMAD canonicalizes
// negative-NaN payloads, which the lane JN certificate distinguishes).
//
// Lane JU coefficient repair (2026-08-31): the negative arm is expm1 stated
// by the shared Cody-Waite statement (fresh_common.h fresh_expm1_cw — the
// production elu/celu/selu family's expm1_cw_clamped numerics).  The previous
// exp_21f(x) - 1 arm carried a +1.72e-3 bias at 0: sem celu(±0) = +1.72e-3
// and POSITIVE outputs for tiny negative inputs (lane JN certificate,
// 16,032/65,536 diverging inputs).  expm1 stated directly is exact at the
// origin and sign-correct.
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

// Fixed dispatch scalars, identical to the values the production dispatch
// sends (sfpu_operations.h celu branch: alpha = 1.0f, 1/alpha = 1.0f) and to
// the golden's alpha=1.0.  Both legs must always receive identical values.
constexpr std::uint32_t FRESH_CELU_ALPHA_BITS       = 0x3f800000u; // 1.0f
constexpr std::uint32_t FRESH_CELU_ALPHA_RECIP_BITS = 0x3f800000u; // 1.0f

template <bool DST_ACCUM_MODE, int ITERATIONS, std::uint32_t ALPHA_BITS = FRESH_CELU_ALPHA_BITS, std::uint32_t ALPHA_RECIP_BITS = FRESH_CELU_ALPHA_RECIP_BITS>
__attribute__((noinline)) void calculate_celu_fresh_cpp()
{
    constexpr float alpha       = __builtin_bit_cast(float, ALPHA_BITS);
    constexpr float alpha_recip = __builtin_bit_cast(float, ALPHA_RECIP_BITS);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Negative branch computed on all lanes (vector select below): the
        // positive lanes' expm1 value is never observed.
        const sfpi::vFloat e = fresh_expm1_cw(v * alpha_recip);
        sfpi::vFloat r       = alpha * e;
        v_if (v >= 0.0f)
        {
            r = v;
        }
        v_endif;
        if constexpr (!DST_ACCUM_MODE)
        {
            // bf16 destination: round to nearest-even before the store
            // truncates (the positive pass-through lanes are already bf16).
            r = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = r;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
