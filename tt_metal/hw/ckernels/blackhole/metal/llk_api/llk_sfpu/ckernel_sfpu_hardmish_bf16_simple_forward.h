// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_hardmish_bf16_replay_window.h"
#include "ckernel_sfpu_hardmish_bf16_simple_algebraic.h"
namespace ckernel::sfpu::bf16::hardmish {
// Rows step DEST through the zero-step hold mode and an INCRWC per replay, as ADDR_MOD_6 is the FPU's;
// a threshold pair that folds its identity store records its own INCRWC.
template <typename Config, int Iterations = 8>
inline void calculate_simple_forward() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    {
        constexpr bool folds = Config::kKind == 1 && Config::kBodySlots == 11;
        constexpr uint32_t step = folds ? 0u : 2u * Config::kRowsPerReplay;
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_UPPER, Config::kSlopeBits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG6, sfpi::SFPLOADI_MOD0_LOWER, Config::kSlopeBits & 65535);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_UPPER, Config::kInterceptBits >> 16);
        TTI_SFPLOADI(p_sfpu::LREG7, sfpi::SFPLOADI_MOD0_LOWER, Config::kInterceptBits & 65535);
        // A row too long for the window records its first 16 slots; a threshold or paired row records whole.
        constexpr bool split = Config::kKind != 1 && Config::kRowsPerReplay != 2;
        constexpr uint32_t head_slots = split ? 16u : Config::kBodySlots;
        auto row_head = [&]() {
            ttpoly::hardmish::
                simple_gated_pair<false, (Config::kRawEqual != 0u), Config::kRawEqual, 0, 0, ADDR_MOD_7, ADDR_MOD_7>();
        };
        auto row_tail = [&]() {
            if constexpr (split) {
                ttpoly::hardmish::simple_bh_gated_joint_row_tail<Config::kRawEqual, ADDR_MOD_7, ADDR_MOD_7>();
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
}  // namespace ckernel::sfpu::bf16::hardmish
