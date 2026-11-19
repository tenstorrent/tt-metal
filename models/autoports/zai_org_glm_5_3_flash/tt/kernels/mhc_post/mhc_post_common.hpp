// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused mHC post-mix (Stage 7a, mhc_post.py is the host side): X'_j = post_j * (y [+ y2]) + sum_i C[i][j] * X_i for
// the 4 streams j, on 64 cores (logical 8 x 8), core k = 8x + y owning column tiles {2k, 2k+1}; no cross-core traffic.
// The per-token coefficients (post_j, C[i][j]) come from mhc_pre's compact coefficient tile (row t = post_0..3 |
// comb_0..15) and are expanded on each core into 20 column-0 tiles (SFPU column broadcast operands).
#pragma once

#ifndef POST_CB0
#define POST_CB0 0  // megakernel: this phase's CB indices shifted past the other phases'
#endif

#include <cstdint>

namespace mpost {

constexpr uint32_t N_STREAMS = 4;
constexpr uint32_t COL_TILES = 128;
constexpr uint32_t COLS_PER_CORE = 2;
constexpr uint32_t X_TILES = N_STREAMS * COLS_PER_CORE;  // index s * 2 + j
constexpr uint32_t N_COEF = 20;                          // post 0..3 | C[i][j] at 4 + 4i + j
constexpr uint32_t FP32_TILE = 4096;

constexpr uint32_t CB_Y = POST_CB0 + 0;     // bf16 sublayer output tiles (2)
constexpr uint32_t CB_Y2 = POST_CB0 + 1;    // bf16 second summand (shared expert), optional (2)
constexpr uint32_t CB_X = POST_CB0 + 2;     // fp32 residual tiles (8)
constexpr uint32_t CB_COEF = POST_CB0 + 3;  // fp32 column-0 coefficient tiles (20)
constexpr uint32_t CB_CROW = POST_CB0 + 4;  // fp32 scratch: the coefficient rows read from mhc_pre's tile
constexpr uint32_t CB_YS = POST_CB0 + 5;    // fp32 y (+ y2) of the current column tile
constexpr uint32_t CB_OUT = POST_CB0 + 6;   // fp32 output tiles (the 4 streams of one column tile)
constexpr uint32_t CB_STK = POST_CB0 + 7;   // fp32 writer scratch: stacked output tile assembly

constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}
// element offset of DST plane p, lane t (the mhc_pre plane map; Stage 14c v2 compact coefficient tile)
constexpr uint32_t plane_off(uint32_t p, uint32_t t) {
    return (p >> 3) * 256u + (((p & 7u) >> 1) * 4u + (t >> 3)) * 16u + 2u * (t & 7u) + (p & 1u);
}

}  // namespace mpost
