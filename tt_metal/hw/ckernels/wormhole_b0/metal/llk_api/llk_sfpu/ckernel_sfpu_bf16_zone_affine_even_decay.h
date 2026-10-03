// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_bf16_zone_affine_even_decay_core.h"
namespace ckernel::sfpu::bf16 {
template <class Config, int Iterations = 32>
inline void calculate_zone_affine_even_decay() {
    static_assert(Iterations == 32, "selected indexed zone body requires the complete tile");
    sfpi::zone_affine_even_decay_init<Config>();
    sfpi::zone_affine_even_decay_tile<Config>(
        [](sfpi::vFloat raw) { return raw; },
        []([[maybe_unused]] sfpi::vFloat raw, [[maybe_unused]] sfpi::vFloat& result) {});
}
}  // namespace ckernel::sfpu::bf16
