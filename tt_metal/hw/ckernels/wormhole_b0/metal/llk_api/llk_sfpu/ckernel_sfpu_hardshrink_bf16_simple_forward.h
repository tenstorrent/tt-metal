// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_hardshrink_bf16_replay_window.h"
#include "ckernel_sfpu_hardshrink_bf16_simple_algebraic.h"
namespace ckernel::sfpu::bf16::hardshrink {
// Rows step DEST through the zero-step hold mode and an INCRWC per replay, as ADDR_MOD_6 is the FPU's;
// a threshold pair that folds its identity store records its own INCRWC.
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    {
        constexpr bool folds = Config::kKind == 1;
        constexpr uint32_t step = folds ? 0u : 2u * Config::kRowsPerReplay;
        {
            constexpr uint32_t pin =
                Config::kBodySlots != 16 ? Config::kThresholdBits : (Config::kComparatorBf16 << 16) ^ 0x80000000u;
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, pin >> 16);
            TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, pin & 65535);
        }
        // A row too long for the window records its first 16 slots; a threshold or paired row records whole.
        constexpr bool split = Config::kKind != 1;
        constexpr uint32_t head_slots = split ? 16u : Config::kBodySlots;
        auto row_head = [&]() { ttpoly::hardshrink::simple_threshold_pair<ADDR_MOD_3, ADDR_MOD_3>(); };
        auto row_tail = [&]() {
            if constexpr (split) {
                ttpoly::hardshrink::simple_wh_gated_row_tail<0, ADDR_MOD_3, ADDR_MOD_3>();
            }
        };
        record_split<0, Config::kBodySlots, head_slots>(row_head, row_tail);
        if constexpr (step) {
            TTI_INCRWC(0, step, 0, 0);
        }
#pragma GCC unroll 8
        for (int row = Config::kRowsPerReplay; row < Iterations; row += Config::kRowsPerReplay) {
            replay_split<0, Config::kBodySlots, head_slots>(row_head, row_tail);
            if constexpr (step) {
                TTI_INCRWC(0, step, 0, 0);
            }
        }
    }
}
}  // namespace ckernel::sfpu::bf16::hardshrink
