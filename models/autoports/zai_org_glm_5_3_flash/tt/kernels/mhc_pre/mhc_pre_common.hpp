// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared constants of the fused mHC "pre" program (Stage 7a, mhc_pre.py is the host side and documents the design).
//
// Core roles (compile-time defines per kernel group):
//   MHC_K       64 K-split cores, logical (x, y) in [0,8) x [0,8); core k = 8x + y owns the global column tiles
//               c in {2k, 2k+1} of all 4 streams (8 x tiles, 8 fn^T tiles) and writes hn tiles 2k, 2k+1.
//   MHC_LEADER  the K cores with y == 0: also gather / sum the partials of their column x and relay the root's
//               messages to it.
//   MHC_ROOT    one extra core: sums the 8 group partials, computes the coefficients and the norm scale.
// Both data-movement RISCs run on every core: "stream" (BRISC, NoC0: DRAM / L1 tensor reads and output writes) and
// "sync" (NCRISC, NoC1: core-to-core partials, semaphores, messages).
#pragma once

#include <cstdint>

namespace mhc {

constexpr uint32_t TILE = 32;
constexpr uint32_t TILE_ELEMS = 1024;
constexpr uint32_t FP32_TILE_BYTES = 4096;
constexpr uint32_t BF16_TILE_BYTES = 2048;
constexpr uint32_t N_STREAMS = 4;
constexpr uint32_t COL_TILES = 128;                      // 4096 / 32 column tiles per stream
constexpr uint32_t COLS_PER_CORE = 2;                    // column tiles per K core
constexpr uint32_t X_TILES = N_STREAMS * COLS_PER_CORE;  // x / fn tiles per K core, index s * 2 + j
constexpr uint32_t GROUP = 8;                            // K cores per leader (one logical column)
constexpr uint32_t N_GROUPS = 8;
constexpr uint32_t N_MIX = 24;        // m values per token: pre 4 | post 4 | comb 16
constexpr uint32_t MSG_FLOATS = 128;  // mailbox: 4 planes x 32 tokens (round 1) / 32 tokens (round 2)

// circular buffers (uniform indices on every core; the first two are defined on every core so that their L1
// addresses are equal everywhere: remote cores write into them)
constexpr uint32_t CB_GATHER = 0;  // fp32 16 pages: round 1 = 8 x (M, S) partials, round 2 = 8 S2 partials
constexpr uint32_t CB_MSG = 1;     // 512 B mailbox (root -> leaders -> members)
constexpr uint32_t CB_X = 2;       // fp32 8 pages, x tiles (matmul operand: default unpack)
constexpr uint32_t CB_X32 = 3;     // alias of CB_X (same allocation) for exact fp32 copy_tile
constexpr uint32_t CB_FN = 4;      // bf16 8 pages, fn^T tiles [32 K rows x 32 values]
constexpr uint32_t CB_W = 5;       // fp32 2 pages, norm weight tiles (row 0)
constexpr uint32_t CB_PART = 6;    // fp32 2 pages, (M, S) partial of this core
constexpr uint32_t CB_PRE = 7;     // fp32 4 pages, pre_s in column 0 (rows = tokens), expanded from the mailbox
constexpr uint32_t CB_H = 8;       // fp32 2 pages, collapsed h tiles (compute scratch)
constexpr uint32_t CB_S2 = 9;      // fp32 1 page, sum_c h^2 partial
constexpr uint32_t CB_INV = 10;    // fp32 1 page, 1/rms(h) in column 0
constexpr uint32_t CB_OUT = 11;    // bf16 (or fp32) 2 pages, hn tiles
constexpr uint32_t CB_XSTK = 12;   // fp32 1 page, stacked output tile assembly (stream RISC)
constexpr uint32_t CB_GSUM = 13;   // fp32 2 pages, group sums (leaders; round 2 uses page 0)
constexpr uint32_t CB_RED = 14;    // fp32 1 page, root: coefficient logits z [tokens x 24]
constexpr uint32_t CB_PLANE = 15;  // fp32 1 page, root: z in the plane layout
constexpr uint32_t CB_COEF = 16;   // fp32 1 page, root: pre | post | comb in the plane layout
constexpr uint32_t CB_SB = 17;     // fp32 2 pages, root: scale row, bias row (row 0 of each)
constexpr uint32_t CB_INVR = 18;   // fp32 1 page, root: 1/rms(h) column
constexpr uint32_t CB_SCR = 19;    // fp32 pages, root: output tile assembly
constexpr uint32_t CB_DBG = 20;    // fp32 2 pages, debug taps packed by the root compute

// semaphores
constexpr uint32_t SEM_G1 = 0;  // round-1 partials arrived (leader: 8 cores, root: 8 groups)
constexpr uint32_t SEM_G2 = 1;  // round-2 partials arrived
constexpr uint32_t SEM_M1 = 2;  // round-1 message (pre) arrived
constexpr uint32_t SEM_M2 = 3;  // round-2 message (1/rms(h)) arrived

// debug taps (DEBUG_TAPS): tile index in the [N_TAPS, 32, 32] fp32 debug tensor
constexpr uint32_t TAP_M_TOTAL = 0;  // root: sum of the 64 M partials
constexpr uint32_t TAP_S_TOTAL = 1;  // root: sum of the 64 S partials
constexpr uint32_t TAP_Z = 2;        // root: logits z (rows tokens, cols 0..23)
constexpr uint32_t TAP_PLANE = 3;    // root: z in the plane layout
constexpr uint32_t TAP_COEF = 4;     // root: SFPU coefficients in the plane layout
constexpr uint32_t TAP_INVR = 5;     // root: 1/rms(h) column
constexpr uint32_t TAP_K0_M = 6;     // K core (0,0): M partial
constexpr uint32_t TAP_K0_S = 7;     // K core (0,0): S partial
constexpr uint32_t TAP_K0_PRE0 = 8;  // K core (0,0): expanded pre_0 tile
constexpr uint32_t TAP_K0_INV = 9;   // K core (0,0): expanded 1/rms(h) tile
constexpr uint32_t N_TAPS = 10;

// element offset of tile element (r, c) in the face-major layout of a 32x32 tile
constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

// Plane layout of one fp32 tile in DST: SFPU address 2p (sfpi::dst_reg[p]) holds rows 4*band .. 4*band+3 of face
// p / 8, band = (p % 8) / 2, in the even (p even) or odd (p odd) columns; its lane 8i + c is row 4*band + i, column
// 2c + parity of that face. "Plane p, token t" is lane t of dst_reg[p], so one SFPU instruction applies to one value
// of all 32 tokens (lane = token) and the 4x4 Sinkhorn is plain element-wise vector code.
constexpr uint32_t plane_off(uint32_t p, uint32_t t) {
    return (p >> 3) * 256u + (((p & 7u) >> 1) * 4u + (t >> 3)) * 16u + 2u * (t & 7u) + (p & 1u);
}

}  // namespace mhc
