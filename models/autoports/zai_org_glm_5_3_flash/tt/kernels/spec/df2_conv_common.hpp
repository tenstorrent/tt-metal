// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 17 DFlash2 in-trace drafter, rung 2 (tt/dflash2_kind.py `conv="fused"`): ONE program per grouped 2-tap dynamic
// convolution call, replacing the 10 stock-op launches of DFlash2Kind._conv. Per row r and channel h of the 32-row tile
// row ([1,1,32,H] bf16):
//   out[r,h] = (base[side,0,h] + cf[r, q(side,0,h)]) * x[r,h] + (base[side,1,h] + cf[r, q(side,1,h)]) * x[r-1,h]
//   q(side,t,h) = side * 2 NG + t * NG + h / G  (NG = H / G groups of G channels; cf = kernel_projection(x) [32, 4 NG])
//   x[-1,:] = 0 (row 0 has no predecessor); fp32 math, bf16 out (the reference tt/dflash2_ref.py DFlash2Ref.conv).
// Every operand is a bf16 tile; the coefficient and the shifted-row tiles are exact matmuls of 0/1 selection tiles
// into the fp32 DST (HiFi4 bf16 x bf16 products are exact), the products / sum are fp32 SFPU ops on DST:
//   coef_t   = CF[q0 / 32] @ E[q0 % 32] + J[2 side + t] @ B[c]      (q0 = side 2 NG + t NG + c GPT, GPT = 32 / G)
//   x_prev   = SH @ X[c]                                            (SH[r, r-1] = 1)
//   out      = x_prev * coef_1 + x * coef_0                         (fp32 SFPU; the reference's two-term sum)
// E[e][j, h] = 1 iff j == e + h / G (the 32 group->channel selection tiles of a column tile whose first group is
// cf column e of its cf tile); J[k][r, j] = 1 iff j == k; B[c] = tile c of the [1,1,32,H] base tensor whose row k =
// base[side = k / 2, tap = k % 2] (rows 4..31 zero) -> J[k] @ B[c] = base row k broadcast to all 32 rows.
// Cores: column tiles [c0, c0 + n) per core (the host balances H / 32 tiles over the grid).
// CBs (bf16, 2-KiB pages): X (x tile), CF (the 2 cf tiles of the column tile: tap 0, tap 1), E (the 2 selection
// tiles), J (2: k0 = 2 side, k1 = 2 side + 1, once), B (base tile), SH (once), ZERO (once), OUT.
// Reader compile-time args: GPT, NG, then the TensorAccessorArgs of x, cf, E, J, B, SH. Runtime args (fixed 10):
// x, cf, E, J, B, SH, out addresses, c0, n, side. Writer: TensorAccessorArgs of out; runtime out, c0, n.
#pragma once

#include <cstdint>

namespace df2conv {

constexpr uint32_t CB_X = 0, CB_CF = 1, CB_E = 2, CB_J = 3, CB_B = 4, CB_SH = 5, CB_ZERO = 6, CB_OUT = 7;
// DF2_CONV_RES (rung 3 M2 prep, the O2 fold of dflash2_intrace.md §7 / §8): the "finish" conv takes the fp32
// all-reduced partial X32 (fp32 tiles, 4-KiB pages) and the bf16 residual RES; x = bf16(X32) is made by a pack /
// unpack pair through CB_X (the packer's fp32 -> bf16 rounding = ttnn.typecast's), out = RES + conv(x) (fp32 DST
// add, packed bf16 = ttnn.add's rounding): the 16 typecast + 16 add launches of the step fold into the 16 programs.
constexpr uint32_t CB_X32 = 8, CB_RES = 9;
constexpr uint32_t TILE_BYTES = 2048;    // bf16 32 x 32
constexpr uint32_t TILE_BYTES32 = 4096;  // fp32 32 x 32

}  // namespace df2conv
