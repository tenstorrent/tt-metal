// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Where the scale rows of super-group G (8 groups of 128) for local column j sit in A3's scale tensor.
// pk = 0: scale_rows layout, s_tiles fp32 tiles per column, super-group G in tile G/4 at rows (G%4)*8.
// pk = 1: A's PackedS (shared tails): per bank, columns in groups of tail_t, each group
//         [tail tile if rem][s_full full tiles per column]; full super-groups as above, tail super-group
//         r = G - 4*s_full of column ti in the group at rows (ti*rem + r)*8 of the group's tail tile.
#pragma once
#include <cstdint>

struct ScaleLoc {
    uint32_t tile;  // fp32 tile index within this bank's scale bytes
    uint32_t row;   // first of the 8 rows
};

template <uint32_t pk, uint32_t s_tiles, uint32_t tail_t, uint32_t s_full, uint32_t rem>
inline ScaleLoc scale_loc(uint32_t j, uint32_t G) {
    if constexpr (pk == 0) {
        return {j * s_tiles + G / 4, (G % 4) * 8};
    } else {
        constexpr uint32_t tpg = tail_t * s_full + (rem ? 1 : 0);
        const uint32_t base = (j / tail_t) * tpg, ti = j % tail_t;
        if (G < 4 * s_full) {
            return {base + (rem ? 1 : 0) + ti * s_full + G / 4, (G % 4) * 8};
        }
        return {base, (ti * rem + G - 4 * s_full) * 8};
    }
}
