// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Shared constants for the qwen_width_norm kernels.
//
// Core (cx, cy) of a gx x gy grid owns width chunk cx (whole blocks of DB tiles) of the tile rows of row group cy.
// Per tile row: partial = sum(x^2) / W over the chunk (fp32), sent to slot cx of CB_GATH on every core of the row
// group; the gx partials are added, rstd = rsqrt(sum + eps), y = x * rstd * gamma (bf8).

#pragma once

#include <stdint.h>

namespace wnorm {

constexpr uint32_t DB = 4;  // tiles per DST pass with fp32 DEST (half sync); chunks are whole blocks of DB

constexpr uint32_t CB_X = 0;       // bf16 x tiles of this core's chunk, 2 rows
constexpr uint32_t CB_SCALER = 2;  // bf16 reduce scaler (1.0 in row 0 of each face)
constexpr uint32_t CB_EPS = 3;     // bf16 eps in column 0
constexpr uint32_t CB_GAMMA = 4;   // bf16 gamma tiles of the chunk (row 0 of faces 0 and 1)
constexpr uint32_t CB_XSQ = 5;     // fp32 x^2 of one row chunk
constexpr uint32_t CB_PART = 6;    // fp32 this core's partial mean(x^2) of one row (compute -> writer)
constexpr uint32_t CB_RSTD = 7;    // fp32 rsqrt(mean + eps)
constexpr uint32_t CB_FUSION = 8;  // fp32 x * rstd
constexpr uint32_t CB_GATH = 9;    // fp32 gx partials of one row, 2 rows (filled remotely by the row group)
constexpr uint32_t CB_ACC = 10;    // fp32 running sum of the partials
constexpr uint32_t CB_OUT = 16;    // bf8 output tiles (compute -> writer)

constexpr uint32_t SEM_RECV = 0;  // partials received on this core (cumulative: gx per row)

// Compile-time args shared by the three kernels (op.py _common_ct): 0 Wt, 1 gx, 2 gy, 3 Rt, 4 maxw, 5 eps fp32 bits,
// 6 1/W fp32 bits. Kernel-specific args follow at CT_NEXT.
constexpr uint32_t CT_NEXT = 7;

// This core's work from its logical coords (cx, cy): width chunk cx in whole blocks of DB tiles (the first
// (Wt / DB) % gx chunks take one more block) of row group cy (the first Rt % gy groups take one more tile row).
// op.py width_chunks / row_groups compute the same split.
struct Work {
    uint32_t c0;    // first tile column
    uint32_t w;     // tiles
    uint32_t row0;  // first tile row
    uint32_t n;     // tile rows
};

inline Work core_work(uint32_t cx, uint32_t cy, uint32_t Wt, uint32_t gx, uint32_t Rt, uint32_t gy) {
    const uint32_t nb = Wt / DB, bb = nb / gx, be = nb % gx;
    const uint32_t rb = Rt / gy, re = Rt % gy;
    Work k;
    k.c0 = (cx * bb + (cx < be ? cx : be)) * DB;
    k.w = (bb + (cx < be ? 1 : 0)) * DB;
    k.row0 = cy * rb + (cy < re ? cy : re);
    k.n = rb + (cy < re ? 1 : 0);
    return k;
}

}  // namespace wnorm
