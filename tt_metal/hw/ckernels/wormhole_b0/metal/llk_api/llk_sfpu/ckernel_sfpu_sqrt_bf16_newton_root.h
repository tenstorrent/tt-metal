// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_sqrt_bf16_replay_window.h"
// Sibling stock wrappers may include another generated config while this
// shared runtime is being parsed. Expose its adapter interface first.
namespace ckernel::sfpu::bf16::sqrt {
template <typename Config>
inline void init_newton_root();
}
// rsqrt includes sqrt before declaring its callback; a composed sqrt config
// can therefore reach this runtime first. These declarations match the
// included stock definitions, without repeating their default arguments.
namespace ckernel::sfpu {
template <bool, int, bool, bool>
inline void calculate_rsqrt();
template <bool, bool, int>
inline void calculate_cube_root();
}  // namespace ckernel::sfpu
#include "ckernel_sfpu_rsqrt.h"
#include "ckernel_sfpu_cbrt.h"
#include "sfpu/ckernel_sfpu_sqrt_bf16_newton_root.h"
namespace ckernel::sfpu::bf16::sqrt {
// The rows read the magic seed and polynomial from the programmable constants that the
// stock sqrt_init (sqrt, rsqrt) or cube_root_init (cbrt) sets, which other ops sharing
// those inits read too; the selected constants must be stock's, so nothing writes them.
// Each row stores through ADDR_MOD_7, which every SFPU init holds at zero; an INCRWC after
// each replayed row steps DEST, as ADDR_MOD_6 belongs to the FPU ops (matmul) between calls.
template <typename Config>
inline void init_newton_root() {
    static_assert(Config::kMagic == 0x5f1110a0u && Config::kC1Bits == 0x401214c9u && Config::kC2Bits == 0x40103626u);
}
}  // namespace ckernel::sfpu::bf16::sqrt
