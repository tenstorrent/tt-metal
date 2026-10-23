// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_cbrt_bf16_replay_window.h"
#include "sfpu/ckernel_sfpu_cbrt_bf16_newton_root_core.h"
#include "sfpu/ckernel_sfpu_cbrt_bf16_newton_root.h"
// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::bf16::cbrt {
template <typename Config>
inline void init_newton_root();
}
namespace ckernel::sfpu::bf16::cbrt {
// The rows read the magic seed and polynomial from the programmable constants that the
// stock sqrt_init (sqrt, rsqrt) or cube_root_init (cbrt) sets, which other ops sharing
// those inits read too; the selected constants must be stock's, so nothing writes them.
// Each row stores through ADDR_MOD_7, which every SFPU init holds at zero; an INCRWC after
// each replayed row steps DEST, as ADDR_MOD_6 belongs to the FPU ops (matmul) between calls.
template <typename Config>
inline void init_newton_root() {
    static_assert(Config::kC0Bits == 0x3fe04c03u && Config::kC1Bits == 0xbfa01f36u && Config::kC2Bits == 0x3f0266d9u);
}
}  // namespace ckernel::sfpu::bf16::cbrt
