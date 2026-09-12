// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SFPU helper for Gemma-style final logit softcapping:
//   y = cap * tanh(x)
// The caller pre-scales x by folding 1 / cap into the vocab weights.

#pragma once

#include <cstdint>

#ifdef TRISC_PACK
#include "ckernel_sfpu_tanh.h"

namespace ckernel {
namespace sfpu {

// Polynomial variant with explicit destination precision. This overload keeps
// the fused Blaze epilogue's rounding; the accurate two-argument API below is unchanged.
template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_logit_softcap(std::uint32_t cap_bits) {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat t = _sfpu_tanh_polynomial_(x);
        sfpi::vFloat cap = sfpi::as<sfpi::vFloat>(sfpi::vInt(cap_bits));
        sfpi::vFloat result = cap * t;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::as<sfpi::vFloat>(sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest));
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS>
inline void calculate_logit_softcap(std::uint32_t cap_bits, std::uint32_t) {
    for (int d = 0; d < ITERATIONS; d++) {
        // ponytail: compute tanh first, load cap after -> cap not live across the
        // register-heavy tanh (was an SFPI reload ICE at -O3). Algebraically identical.
        sfpi::vFloat t = _sfpu_tanh_fp32_accurate_(sfpi::dst_reg[0]);
        sfpi::vFloat cap = sfpi::as<sfpi::vFloat>(sfpi::vInt(cap_bits));
        sfpi::dst_reg[0] = cap * t;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
#endif  // TRISC_PACK
