// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_hardswish_bf16_simple_forward.h"
#include "sfpu/ckernel_sfpu_hardswish_bf16_simple_algebraic.h"
namespace ckernel::sfpu::bf16::hardswish {
template <typename Config>
inline void init_simple_forward() {
    // Only the replay's advance mode needs programming. A threshold pair that
    // folds its identity store (always on WH, at 11 slots on BH) advances with
    // INCRWC and never reads it.
    constexpr bool folds = Config::kKind == 1;
    if constexpr (Config::kRowsPerReplay && !folds) {
        addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    }
}
}  // namespace ckernel::sfpu::bf16::hardswish
