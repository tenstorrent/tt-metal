// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_bf16_hyperbolic_exp_core.h"
#include "sfpu/ckernel_sfpu_bf16_hyperbolic_exp.h"
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_hyperbolic_exp() {
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[4]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, Config::kScaledCoefficientBits[3]);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    sfpu_reciprocal_init<false>();
}
}  // namespace ckernel::sfpu::bf16
