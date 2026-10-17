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
#include "ckernel_sfpu_bf16_mirrored_terminals.h"
#include "ckernel_sfpu_bf16_rational_parity_core.h"
}  // namespace sfpi
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_rational_parity() {
    ckernel::sfpu::sfpu_reciprocal_init<false>();
}

// Flattened so the core's coefficients are compile-time constants: unit
// coefficients then come from the constant register instead of a load per row.
template <typename Config, int Iterations = 32>
__attribute__((flatten)) inline void calculate_rational_parity() {
    static_assert(Iterations == 32, "selected scalar parity requires a complete tile");
    static_assert(Config::kPinTop && !Config::kRawEarly);
    sfpi::mirrored_parity_rational_tile<
        Config::kNumDegree,
        Config::kDenDegree,
        Config::kBoundBits,
        Config::kRawClass,
        Config::kPinTop,
        Config::kRawEarly>(Config::kNumerator, Config::kDenominator);
}
}  // namespace ckernel::sfpu::bf16
