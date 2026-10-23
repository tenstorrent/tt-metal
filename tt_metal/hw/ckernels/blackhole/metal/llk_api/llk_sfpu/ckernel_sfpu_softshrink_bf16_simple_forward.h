// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_softshrink_bf16_replay_window.h"
#include "ckernel_sfpu_softshrink_bf16_simple_algebraic.h"
namespace ckernel::sfpu::bf16::softshrink {
template <typename Config>
inline void init_simple_forward() {}
// The rows load and store through the hold mode (ADDR_MOD_7, ADDR_MOD_3 past WH's SFPU base), which
// every SFPU init leaves at zero step, and an INCRWC after each replay steps DEST: ADDR_MOD_6 belongs
// to the FPU ops (matmul) a kernel may run between this init and its call. A threshold pair that folds
// its identity store (always on WH, at 11 slots on BH) records its own INCRWC.
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y;
        y = ttpoly::softshrink::simple_threshold_softshift<true>(x, __builtin_bit_cast(float, Config::kThresholdBits));
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu::bf16::softshrink
