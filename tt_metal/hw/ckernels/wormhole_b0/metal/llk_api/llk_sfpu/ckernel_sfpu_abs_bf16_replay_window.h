// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace ckernel::sfpu::bf16::abs {
#define TTPOLY_SFPU_REPLAY_SLOTS 16
inline constexpr std::uint32_t kSfpuReplaySlots = TTPOLY_SFPU_REPLAY_SLOTS;
template <std::uint32_t Start, std::uint32_t Slots>
inline constexpr bool replay_fits = Start + Slots <= kSfpuReplaySlots;
template <std::uint32_t Start, std::uint32_t Slots, typename Row>
inline void record_row(Row&& row) {
    if constexpr (replay_fits<Start, Slots>) {
        TTI_REPLAY(Start, Slots, 1, 1);
        row();
    }
    if constexpr (!replay_fits<Start, Slots>) {
        row();
    }
}
// One replay of the row recorded at Start, or the row issued where it did not fit the window.
template <std::uint32_t Start, std::uint32_t Slots, typename Row>
inline void replay_row(Row&& row) {
    if constexpr (replay_fits<Start, Slots>) {
        TTI_REPLAY(Start, Slots, 0, 0);
    } else {
        row();
    }
}
}  // namespace ckernel::sfpu::bf16::abs
