// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_abs_bf16_replay_window.h"

namespace ckernel::sfpu::bf16::abs {
constexpr uint32_t kAbsHold = ADDR_MOD_3;

template <typename Config, int Iterations = 8>
inline void calculate_abs_value() {
    static_assert(Iterations > 0 && Iterations % 2 == 0);
    auto row_body = [&]() {
        TTI_SFPLOAD(p_sfpu::LREG0, Config::kLoadFormat, kAbsHold, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, Config::kLoadFormat, kAbsHold, 2);
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, Config::kAbsMode);
        TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, Config::kAbsMode);
        TTI_SFPSTORE(p_sfpu::LREG0, Config::kStoreFormat, kAbsHold, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, Config::kStoreFormat, kAbsHold, 2);
    };
    record_row<0, Config::kBodySlots>(row_body);
    // DEST is addressed only through the hold mode, which every SFPU init leaves at zero and no FPU
    // op sets; the counter steps past the pair outside the record, so the replay keeps its length.
    TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 4
    for (int pair = 1; pair < Iterations / 2; ++pair) {
        replay_row<0, Config::kBodySlots>(row_body);
        TTI_INCRWC(0, 4, 0, 0);
    }
}
}  // namespace ckernel::sfpu::bf16::abs
