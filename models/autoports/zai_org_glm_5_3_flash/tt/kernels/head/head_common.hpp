// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14b model head (tt/mk_head.py MkHead), the final norm on the 64 owner cores: h = RMSNorm(mean of the 4
// hyper-connection streams of the residual) * w, T <= 32 token rows (row t in the stacked residual of 4-row chunk t /
// 4, local row t % 4). Owner k holds column tiles c0 = 2k, 2k + 1 of the hidden.
//   BRISC: the stacked residual [T, 1, 4, 4096] (token t, column tile c at page t * 128 + c, stream s in row s)
//          re-laid out row by row into 8 flat fp32 tiles (stream s, column j: row t = token t), the 2 norm-weight row
//          tiles in; later the 2 bf16 h tiles out (pages c0, c0 + 1 of h [1, 1, T, 4096]) + the readers signalled.
//   TRISC: m_j = 0.25 * (x_0j + x_1j + x_2j + x_3j); P = rowsum(m_0^2 + m_1^2) (this owner's partial sum of squares,
//          column 0) -> CB_P; then G = the 64 owners' partials (gathered by the NCRISC: owner k in column k % 32 of
//          tile k / 32) -> inv = rsqrt(rowsum(G0 + G1) / 4096 + eps) (column 0) -> h_j = m_j * inv * w_j -> CB_OUT.
//   NCRISC: P column 0 (T values) -> a 16-B record into slot k of EVERY owner's landing area + their HEAD_SEM; waits
//          for all 64, lays the records into the 2 G tiles.
#pragma once

#include <cstdint>

#ifndef HEAD_CB0
#define HEAD_CB0 0
#endif

namespace head {

constexpr uint32_t TILE = 32;
constexpr uint32_t FP32_TILE_BYTES = 4096;
constexpr uint32_t N_STREAMS = 4;
constexpr uint32_t COL_TILES = 128;
constexpr uint32_t COLS = 2;  // column tiles per owner
constexpr uint32_t N_OWNERS = 64;
constexpr uint32_t REC_BYTES = 128;  // one owner's partials: T <= 32 fp32
constexpr uint32_t CHUNK_ROWS = 4;   // a stacked residual tensor holds 4 token rows (B > 4: one tensor per 4-row chunk)

constexpr uint32_t CB_X = HEAD_CB0 + 0;     // fp32 8 tiles: flat x, index s * 2 + j
constexpr uint32_t CB_W = HEAD_CB0 + 1;     // fp32 2 tiles: norm weight rows
constexpr uint32_t CB_M = HEAD_CB0 + 2;     // fp32 2 tiles: the stream means
constexpr uint32_t CB_P = HEAD_CB0 + 3;     // fp32 1 tile: this owner's partial sum of squares (column 0)
constexpr uint32_t CB_G = HEAD_CB0 + 4;     // fp32 2 tiles: the 64 owners' partials
constexpr uint32_t CB_OUT = HEAD_CB0 + 5;   // bf16 2 tiles: h
constexpr uint32_t CB_LAND = HEAD_CB0 + 6;  // 64 x 16 B landing area (raw L1, written by every owner's NCRISC)
constexpr uint32_t CB_REC = HEAD_CB0 + 7;   // 16 B (+ alignment) scratch: this owner's record
constexpr uint32_t EMB_CB = HEAD_CB0 + 8;   // MkEmbed: one fp32 tile + 256 B + 128 B of staging

// element offset of tile element (r, c) in the face-major layout of a 32x32 tile
constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace head
