// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
namespace ckernel::sfpu::bf16::rsqrt {
#define TTPOLY_SFPU_REPLAY_SLOTS 16
inline constexpr std::uint32_t kSfpuReplaySlots = TTPOLY_SFPU_REPLAY_SLOTS;
template <std::uint32_t Start, std::uint32_t Slots>
inline constexpr bool replay_fits = Start + Slots <= kSfpuReplaySlots;
template <std::uint32_t Start, std::uint32_t Slots, std::uint32_t Head, typename HeadRow, typename TailRow>
inline void record_split(HeadRow&& head, TailRow&& tail) {
    static_assert(Head > 0 && Head <= Slots);
    if constexpr (replay_fits<Start, Slots>) {
        TTI_REPLAY(Start, Slots, 1, 1);
        head();
        tail();
    }
    if constexpr (!replay_fits<Start, Slots> && replay_fits<Start, Head>) {
        TTI_REPLAY(Start, Head, 1, 1);
        head();
        tail();
    }
    if constexpr (!replay_fits<Start, Slots> && !replay_fits<Start, Head>) {
        head();
        tail();
    }
}
template <std::uint32_t Start, std::uint32_t Slots, std::uint32_t Head, typename HeadRow, typename TailRow>
inline void replay_split(HeadRow&& head, TailRow&& tail) {
    if constexpr (replay_fits<Start, Slots>) {
        TTI_REPLAY(Start, Slots, 0, 0);
    } else if constexpr (replay_fits<Start, Head>) {
        TTI_REPLAY(Start, Head, 0, 0);
        tail();
    } else {
        head();
        tail();
    }
}
}  // namespace ckernel::sfpu::bf16::rsqrt
