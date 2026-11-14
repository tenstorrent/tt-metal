// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_hardmish_bf16_min_max.h"
#include "ckernel_sfpu_hardmish_bf16_simple_algebraic.h"
namespace ckernel::sfpu::bf16::hardmish {
// Rows step DEST through the zero-step hold mode and an INCRWC per replay, as ADDR_MOD_6 is the FPU's;
// a threshold pair that folds its identity store records its own INCRWC.
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
#pragma GCC unroll 32
    for (int row = 0; row < Iterations; ++row) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y;
        y = ttpoly::hardmish::simple_gated_product(
            x, __builtin_bit_cast(float, Config::kInterceptBits), __builtin_bit_cast(float, Config::kSlopeBits));
        y = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0] = y;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu::bf16::hardmish
