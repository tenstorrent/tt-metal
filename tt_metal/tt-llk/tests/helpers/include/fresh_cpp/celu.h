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
// alpha and 1/alpha arrive as raw fp32 bits exactly as the production
// dispatch sends them (both 1.0f; the golden uses alpha = 1.0).  exp is the
// shared exponent/mantissa-recombination statement (fresh_common.h).
#include <cstdint>

#include "fresh_cpp/fresh_common.h"

namespace ckernel::sfpu
{

// Fixed dispatch scalars, identical to the values the production dispatch
// sends (sfpu_operations.h celu branch: alpha = 1.0f, 1/alpha = 1.0f) and to
// the golden's alpha=1.0.  Both legs must always receive identical values.
constexpr std::uint32_t FRESH_CELU_ALPHA_BITS       = 0x3f800000u; // 1.0f
constexpr std::uint32_t FRESH_CELU_ALPHA_RECIP_BITS = 0x3f800000u; // 1.0f

template <bool DST_ACCUM_MODE, int ITERATIONS>
__attribute__((noinline)) void calculate_celu_fresh_cpp(const std::uint32_t alpha_bits, const std::uint32_t alpha_recip_bits)
{
    const sfpi::vFloat alpha       = Converter::as_float(alpha_bits);
    const sfpi::vFloat alpha_recip = Converter::as_float(alpha_recip_bits);
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        // Negative branch computed on all lanes (vector select below): the
        // positive lanes' exp value is never observed.
        const sfpi::vFloat e = fresh_exp(v * alpha_recip);
        sfpi::vFloat r       = alpha * (e - 1.0f);
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
