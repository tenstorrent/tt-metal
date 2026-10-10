// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_bf16_newton_root.h"
// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::bf16 {
template <typename Config>
inline void init_newton_root();
}
namespace ckernel::sfpu::bf16 {
// The rows read the magic seed and polynomial from the programmable constants that the
// stock sqrt_init (sqrt, rsqrt) or cube_root_init (cbrt) sets, which other ops sharing
// those inits read too; the selected constants must be stock's, so nothing writes them.
template <typename Config>
inline void init_newton_root() {
    static_assert(Config::kMagic == 0x5f1110a0u && Config::kC1Bits == 0x401214c9u && Config::kC2Bits == 0x40103626u);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}
}  // namespace ckernel::sfpu::bf16
