// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared constants of the fused KDA "pre" program (Stage 7a, kda_pre.py is the host side and documents the design).
// 64 cores (logical 8 x 8), core k = 8x + y, no cross-core traffic:
//   conv     qkv column tiles [12k, 12k + 12) of the 768 (q | k | v, 256 tiles each)
//   gates    g / gate_out column tiles [4k, 4k + 4) of the 256 (f_b / g_b projections K = 128 computed here)
//   beta     tile k of the 2 (cores 0, 1)
// ROWS mode (B <= ROWS_MAX): the B valid rows of the core's 12 column tiles are packed into one tile per token
// (packed row r = column tile c0 + r, row t), the conv runs on B packed tiles and the reader also shifts the conv
// history rows of the active users in place (state.C row writes). TILES mode: whole tiles, history left to the chain.
#pragma once

#ifndef KDA_CB0
#define KDA_CB0 0  // megakernel: this phase's CB indices shifted past the other phases'
#endif

#include <cstdint>

namespace kdap {

// Head count of the layer shard (Stage 14 TP8: 8 heads per chip; default = the whole layer's 64). Every active core
// owns one head's worth of columns (12 conv tiles = 3 x 4, 4 g / gate_out tiles), so H = 64 is exactly the original
// 64-core layout; cores k >= H of the program's core list skip the phase (megakernel owners beyond the shard's heads).
#ifndef KDA_HEADS
#define KDA_HEADS 64
#endif
// KDA_SPREAD (Stage 14 TP: a chip's few heads spread over more cores): cores per head's worth of columns (1, 2, 4):
// a core owns 12 / SPREAD conv tiles and 4 / SPREAD g / gate_out tiles; 1 = the original layout.
#ifndef KDA_SPREAD
#define KDA_SPREAD 1
#endif
static_assert(KDA_SPREAD == 1 || KDA_SPREAD == 2 || KDA_SPREAD == 4, "KDA_SPREAD divides 12 and 4");
// Stage 14 KDA_RSPLIT (GLM53_KDA_RSPLIT, ROWS mode): RSPLIT row groups of the N_CORES column cores — core kk = column
// core kk % N_CORES, row group kk / N_CORES; group 0 also computes g / gate_out / beta (whole-row tiles), every group
// convolves its row pairs [p0, p1) (an equal split of the B / 2 pairs; a group may have none)
#ifndef KDA_RSPLIT
#define KDA_RSPLIT 1
#endif
constexpr uint32_t RSPLIT = KDA_RSPLIT;
constexpr uint32_t QKV_TILES = 12 * KDA_HEADS;  // 3 * H * 128 / 32
constexpr uint32_t W_TILES = 4 * KDA_HEADS;     // H * 128 / 32
constexpr uint32_t CONV_PER_CORE = 12 / KDA_SPREAD;
constexpr uint32_t G_PER_CORE = 4 / KDA_SPREAD;
constexpr uint32_t K_TILES = 4;                       // f_b / g_b: K = 128
constexpr uint32_t N_CORES = KDA_HEADS * KDA_SPREAD;  // active column cores (x RSPLIT row groups)
// KDA_PACK (7a, ROWS mode): user rows per packed tile, even. Packed row KDA_PACK * r + p = column tile c0 + r of user
// row KDA_PACK * u + p (unit u of the core's pairs); 2 = the original pair layout. At SPREAD 4 (3 conv tiles) up to 10
// users share one tile: 4 units at B = 32 instead of 16 row pairs (the pair loop is latency bound: one read / write
// round trip each); with RSPLIT 2, 2 units per row group instead of 8 pairs.
#ifndef KDA_PACK
#define KDA_PACK 2
#endif
static_assert(KDA_PACK % 2 == 0 && KDA_PACK * CONV_PER_CORE <= 32, "KDA_PACK: even, KDA_PACK * conv tiles <= 32");
// user-row pairs per packed tile: pair p0 + UP * u + pp of unit u -> packed row pair UP * r + pp of column tile c0 + r
constexpr uint32_t UP = KDA_PACK / 2;
// row group rg's pair range of `pairs` user-row pairs
inline void rsplit_range(uint32_t pairs, uint32_t rg, uint32_t& p0, uint32_t& p1) {
    const uint32_t per = (pairs + RSPLIT - 1) / RSPLIT;
    p0 = rg * per < pairs ? rg * per : pairs;
    p1 = p0 + per < pairs ? p0 + per : pairs;
}
constexpr uint32_t BETA_TILES = (KDA_HEADS + 31) / 32;  // beta [B, H] column tiles (cores 0 .. BETA_TILES-1)
constexpr uint32_t HIST = 3;                            // conv history rows t-3 .. t-1
constexpr uint32_t BF16_TILE = 2048;
constexpr uint32_t FP32_TILE = 4096;

// circular buffers
constexpr uint32_t CB_H0 = KDA_CB0 + 0;  // bf16 conv history row 0 (packed per token, or whole tiles)
constexpr uint32_t CB_H1 = KDA_CB0 + 1;
constexpr uint32_t CB_H2 = KDA_CB0 + 2;
constexpr uint32_t CB_Q = KDA_CB0 + 3;  // bf16 qkv
constexpr uint32_t CB_T = KDA_CB0 + 4;  // taps j = 0..3 (hist rows, current row): ROWS 4 packed bf16 tiles (resident);
                                        // TILES 4 fp32 row-0 tiles per column tile
constexpr uint32_t CB_CONV = KDA_CB0 + 5;   // fp32 silu(conv) per unit
constexpr uint32_t CB_FA = KDA_CB0 + 6;     // bf16 f_a tiles (4)
constexpr uint32_t CB_GA = KDA_CB0 + 7;     // bf16 g_a tiles (4)
constexpr uint32_t CB_WFB = KDA_CB0 + 8;    // bf16 w_fb tiles [kt][n] (16)
constexpr uint32_t CB_WGB = KDA_CB0 + 9;    // bf16 w_gb tiles (16)
constexpr uint32_t CB_DT = KDA_CB0 + 10;    // fp32 dt_bias row-0 tiles (4)
constexpr uint32_t CB_RATE = KDA_CB0 + 11;  // fp32 decay-rate row-0 tiles (4)
constexpr uint32_t CB_ACT = KDA_CB0 + 12;   // fp32 active (col 0) and lower_bound * active (col 0)
constexpr uint32_t CB_G = KDA_CB0 + 13;     // fp32 g output tiles
constexpr uint32_t CB_GO = KDA_CB0 + 14;    // fp32 gate_out output tiles
constexpr uint32_t CB_B = KDA_CB0 + 15;     // bf16 b tile (beta cores)
constexpr uint32_t CB_BETA = KDA_CB0 + 16;  // fp32 beta output tile
constexpr uint32_t CB_SCR = KDA_CB0 + 17;   // reader scratch (mixed-pair history rows), 384 B

constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace kdap
