// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_signed_abs_affine.h"

namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_signed_abs_affine() {
    sfpi::signed_abs_affine_init<Config>();
}

template <typename Config, int Iterations = 8>
inline void calculate_signed_abs_affine() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    sfpi::signed_abs_affine_pin_coefficients<Config>();
    TTI_REPLAY(0, Config::kBodySlots, 1, 1);
    sfpi::signed_abs_affine_body<Config, ADDR_MOD_7, ADDR_MOD_6>();
#pragma GCC unroll 8
    for (int row = 1; row < Iterations; ++row) {
        TTI_REPLAY(0, Config::kBodySlots, 0, 0);
    }
    sfpi::signed_abs_affine_drain();
    // No deployment SETRWC: the upstream LLK owns face traversal/counters.
}
}  // namespace ckernel::sfpu::bf16
