// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7a prefill mHC post-mix (lever A, tt/mhc_pre_pf.py mhc_post_prefill_7a): the arithmetic of Stage 16's
// kernels/prefill/mhc_post_pf_* (out_s = post_s * y + sum_i C[i][s] x_i per token row, exact fp32 SFPU) with ONE SFPU
// pass per output tile instead of 5 multiplies + 4 adds (each with its init, the column broadcast rebuilt per call):
// y, x_0..3 and the output stream's two coefficient tiles sit in DST together (full-sync fp32 DST: 8 tiles) and a
// custom pass broadcasts each per-row coefficient once per 4-row band and FMAs it into 4 accumulator registers.
#pragma once

#include <cstdint>

namespace mpf7 {

constexpr uint32_t N_STREAMS = 4;
constexpr uint32_t FP32_TILE = 4096;
// per row tile, per output stream s: tile K_s = post_s | C[0][s] | C[1][s] | C[2][s] in columns 0 / 1 / 16 / 17 (lane 0
// of the 4 SFPLOAD slots of a band: a slot holds the even (+0) or odd (+2) columns of a face, lane c = column 2c +
// odd), tile L_s = C[3][s] in column 0 (every other element zero: the broadcast masks multiply them by 0)
constexpr uint32_t N_COEF = 2 * N_STREAMS;

constexpr uint32_t CB_Y = 0;     // y tile of the unit (2 pages)
constexpr uint32_t CB_X = 1;     // the 4 stream tiles of the unit (8 pages)
constexpr uint32_t CB_COEF = 2;  // the 8 coefficient tiles of the current row tile
constexpr uint32_t CB_RAW = 3;   // reader scratch: the raw post / comb tiles (2 pages)
constexpr uint32_t CB_OUT = 4;   // the 4 output tiles of the unit (8 pages)

// DST tiles of a pass
constexpr uint32_t D_Y = 0;
constexpr uint32_t D_X0 = 1;  // .. D_X0 + 3
constexpr uint32_t D_K = 5;
constexpr uint32_t D_L = 6;
constexpr uint32_t D_OUT = 7;

constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

}  // namespace mpf7
