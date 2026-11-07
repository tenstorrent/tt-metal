// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_gelu_bf16_zone_affine_even_decay_core.h"
namespace ckernel::sfpu::bf16::gelu {

template <class Config, int Iterations = 32>
inline void calculate_zone_affine_even_decay() {
    static_assert(Iterations == 32, "selected indexed zone body requires the complete tile");
    auto prepare = [](sfpi::vFloat raw) __attribute__((always_inline)) { return raw; };
    auto finalize = []([[maybe_unused]] sfpi::vFloat raw, [[maybe_unused]] sfpi::vFloat& result)
                        __attribute__((always_inline)) {};
    ttpoly::gelu::zone_affine_even_decay_tile<Config, decltype(prepare), decltype(finalize), false>(prepare, finalize);
}
}  // namespace ckernel::sfpu::bf16::gelu
