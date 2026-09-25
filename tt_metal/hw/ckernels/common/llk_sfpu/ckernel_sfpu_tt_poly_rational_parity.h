// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
namespace sfpi {
#include "ckernel_sfpu_tt_poly_mirrored_terminals.h"
#include "ckernel_sfpu_tt_poly_rational_parity_core.h"
}  // namespace sfpi
namespace ckernel::sfpu::ttpoly {
template <typename Config, int Iterations = 32>
inline void calculate_rational_parity() {
    static_assert(Iterations == 32, "selected scalar parity requires a complete tile");
#if defined(ARCH_BLACKHOLE)
    static_assert(!Config::kPinTop && Config::kRawEarly);
#elif defined(ARCH_WORMHOLE)
    static_assert(Config::kPinTop && !Config::kRawEarly);
#else
#error "selected mirrored parity requires BH or WH"
#endif
    // Numerical constants only: upstream caller retains SFPU counters/address modes.
    ckernel::sfpu::sfpu_reciprocal_init<false>();
    sfpi::mirrored_parity_rational_tile<
        Config::kNumDegree,
        Config::kDenDegree,
        Config::kBoundBits,
        Config::kRawClass,
        Config::kPinTop,
        Config::kRawEarly>(Config::kNumerator, Config::kDenominator);
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_MIRRORED_PARITY_RATIONAL_V1 1
