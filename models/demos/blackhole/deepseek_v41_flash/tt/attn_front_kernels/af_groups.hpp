// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Column groups of the grouped weight layout (see tt/attn_front.py shuffle): the weights of a bank shard are stored as
// groups of CT columns, inside a group k-major (for every k row the CT tiles of the group), so one custom_mm_block(kt,
// ct) call reuses the activation tile over CT columns (7.8 ns per tile product at ct = 7, 8.9 at 2, 11.4 at 1).  The
// streamed blocks are always 80 tiles.  The output tiles of a group are [row tile A: CT tiles][row tile B: CT tiles].
#pragma once
#include <cstdint>

namespace af {
constexpr uint32_t NG1 = 4;  // wqkv: 7 columns per bank core
constexpr uint32_t GRP1[NG1] = {2, 2, 2, 1};
constexpr uint32_t NG2 = 8;  // wq_b: 16 columns (one head) per bank core
constexpr uint32_t GRP2[NG2] = {2, 2, 2, 2, 2, 2, 2, 2};

// index of the output tile (as pushed by the compute kernel) of column `col`, row tile `m`
template <uint32_t NG>
constexpr uint32_t out_idx(const uint32_t (&grp)[NG], uint32_t MT, uint32_t col, uint32_t m) {
    uint32_t start_col = 0, start_tile = 0;
    for (uint32_t g = 0; g < NG; ++g) {
        if (col < start_col + grp[g]) {
            return start_tile + m * grp[g] + (col - start_col);
        }
        start_col += grp[g];
        start_tile += grp[g] * MT;
    }
    return 0;
}
}  // namespace af
