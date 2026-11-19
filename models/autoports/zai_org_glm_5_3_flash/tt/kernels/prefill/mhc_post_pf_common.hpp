// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 prefill mHC mixes (tt/prefill_tp.py mhc_post_prefill / mhc_pre_prefill) over T prefill rows of the FLAT
// residual X [1, 1, T, 4 * 4096] fp32 (stream i = column tiles [i CT, (i + 1) CT)), exact fp32 SFPU products and sums
// with column broadcasts of the per-row coefficients (the composite ops' fp32 arithmetic in ONE program):
//   post (default): X'_s = post_s * y + sum_i C[i][s] * X_i -> the flat residual [1, 1, T, 4 * 4096] fp32
//                   (y [T, 4096] bf16 / fp32, post [T, 4], comb [T, 16]: C[i][s] at column 4 i + s)
//   MPP_PRE:        h = sum_i pre_i * X_i -> [1, 1, T, 4096] fp32 (pre [T, 4])
// Work unit = (row tile r, column tile c); a core owns a contiguous unit range (row-major), so its coefficient tiles
// change only when its row tile does. All tensors TILE interleaved DRAM.
#pragma once

#include <cstdint>

namespace mpp {

constexpr uint32_t N_STREAMS = 4;
#ifdef MPP_PRE
constexpr uint32_t N_COEF = 4;  // pre_0..3
constexpr uint32_t N_OUT = 1;
#else
constexpr uint32_t N_COEF = 20;  // post_0..3 | C[i][s] at 4 + 4 i + s
constexpr uint32_t N_OUT = 4;
#endif
constexpr uint32_t FP32_TILE = 4096;

constexpr uint32_t CB_Y = 0;     // y tile of the unit (2 pages)
constexpr uint32_t CB_X = 1;     // the 4 stream tiles of the unit (8 pages)
constexpr uint32_t CB_COEF = 2;  // fp32 column-0 coefficient tiles of the current row tile (20 pages)
constexpr uint32_t CB_RAW = 3;   // reader scratch: the raw post / comb tiles (2 pages)
constexpr uint32_t CB_OUT = 4;   // the 4 output tiles of the unit (8 pages)

constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace mpp
