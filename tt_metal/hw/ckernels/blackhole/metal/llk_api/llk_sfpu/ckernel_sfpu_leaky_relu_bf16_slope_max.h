// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_leaky_relu_bf16_slope_max.h"

namespace ckernel::sfpu::bf16::leaky_relu {
template <typename Config>
inline void init_slope_max() {
    sfpi::vConstFloatPrgm0 = __builtin_bit_cast(float, Config::kSlopeBits);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16::leaky_relu
