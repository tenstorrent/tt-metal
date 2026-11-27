// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 batched prefill (tt/prefill_tp.py EpMoe, GLM53_PF_MOE_COMBINE=tiles): the EP combine bounded by the LOCAL
// pairs. y[t] = sum_j w[j][t] * out[dest[j C + t]] for the chunk's C tokens, where out = moe_fused_swiglu's output
// [CAP, H] bf16 TILE (the expert rows in dispatch order), dest = the plan's dispatch row of pair p = j C + t, w = the
// pair's routing weight x local mask ([K, C] fp32 ROW_MAJOR, one page: 0 for another chip's pair). Work unit = (token
// row tile r, column tile c): the reader assembles, per slot j, the 32 token rows of column tile c into one bf16 tile
// (local pairs' rows read from out's tiles, the others zero), and the 32 weights of each slot into column-0 fp32
// tiles; the compute accumulates the K weighted tiles with exact fp32 SFPU products / sums; the writer stores y[r, c].
#pragma once

#include <cstdint>

namespace mcomb {

constexpr uint32_t BF16_TILE = 2048;
constexpr uint32_t FP32_TILE = 4096;

constexpr uint32_t CB_S = 0;    // the unit's K assembled slot tiles (bf16, 2 K pages)
constexpr uint32_t CB_W = 1;    // the row tile's K column-0 weight tiles (fp32, K pages)
constexpr uint32_t CB_IDX = 2;  // reader scratch: the row tile's dest / w words + 64-B staging (1 page)
constexpr uint32_t CB_Y = 3;    // output tiles (bf16, 2 pages)

constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace mcomb
