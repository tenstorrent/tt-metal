// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_bf16_newton_reciprocal.h"

namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_newton_reciprocal() {
    static_assert(Config::kC0Bits == 0x3ea57ebbu && Config::kC1Bits == 0x3fba2e90u &&
                  Config::kC2Bits == 0x4007c1f2u && Config::kMagic == 0x7f800000u);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16
