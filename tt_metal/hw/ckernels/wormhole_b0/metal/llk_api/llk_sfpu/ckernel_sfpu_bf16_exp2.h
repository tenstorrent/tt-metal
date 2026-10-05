// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_bf16_exp2_paired.h"
#include "sfpu/ckernel_sfpu_bf16_exp2.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_exp2() {
    static_assert(Config::kDegree == 2u || Config::kDegree == 3u);
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kMultiplierBits);
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[Config::kDegree]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[Config::kDegree - 1]);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16
