// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill mHC "pre" (Stage 7a lever A, tt/mhc_pre_pf.py documents the design): the decode program's arithmetic
// (kernels/mhc_pre) over T rows of the flat residual x [T, 16384] fp32, in row blocks of 32 tokens.
//
// Roles (compile-time defines per kernel group):
//   PF_MEMBER  G groups x P member cores; member j of a group owns the global column tiles [j CPC, (j + 1) CPC) of
//              all 4 streams (CPC = 128 / P): X_TILES = 4 CPC x tiles per row block, fn^T and the norm weight of its
//              columns resident for the whole program.
//   PF_ROOT    one core per group: sums the P partials of a row block, computes its coefficients (the decode root's
//              SFPU chain) and 1/rms(h), writes post / comb of the row block.
// PF_FPU: S = sum x^2 and S2 = sum h^2 as diag(x x^T) / diag(h h^T) matmuls (tf32 operands, fp32 DST; the root masks
// the diagonal with an identity tile), the collapse h_j = sum_s diag(pre_s) @ x_{s,j} on the FPU (CB_PRE holds the
// diagonal tiles) -- the precision of the op chain's FPU reduce / rms_norm, instead of the exact-fp32 SFPU passes.
// Group g handles the row blocks g, g + G, g + 2G, ... in order. Per row block, member: (M, S) partial -> root ->
// pre message -> h (exact fp32 SFPU collapse) -> S2 partial -> root -> 1/rms(h) message -> hn = h * inv * w.
// BRISC (stream, NoC0): DRAM reads / writes. NCRISC (sync, NoC1): core-to-core partials, semaphores, messages.
#pragma once

#include <cstdint>

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp"

namespace mhcpf {

using mhc::BF16_TILE_BYTES;
using mhc::COL_TILES;
using mhc::FP32_TILE_BYTES;
using mhc::MSG_FLOATS;
using mhc::N_MIX;
using mhc::N_STREAMS;
using mhc::plane_off;
using mhc::TILE;
using mhc::tile_off;

constexpr uint32_t ROW_TILES = N_STREAMS * COL_TILES;  // 512 x tiles per row block of the flat residual

// circular buffers (uniform indices on every core; GATHER / MSG are defined on every core so that their L1 addresses
// are equal everywhere: remote cores write into them)
constexpr uint32_t CB_GATHER = 0;  // fp32 2P pages (root): round 1 = P x (M, S); round 2 = P S2 in pages 0..P-1
constexpr uint32_t CB_MSG = 1;     // 512 B mailbox (root -> members)
constexpr uint32_t CB_X = 2;       // fp32 XBUF x X_TILES pages, x tiles of a row block (matmul operand)
constexpr uint32_t CB_X32 = 3;     // alias of CB_X (same allocation) for exact fp32 copy_tile
constexpr uint32_t CB_FN = 4;      // bf16 X_TILES pages, fn^T tiles (resident)
constexpr uint32_t CB_W = 5;       // fp32 CPC pages, norm weight tiles, row 0 (resident)
constexpr uint32_t CB_PART = 6;    // fp32 2 pages, (M, S) partial
constexpr uint32_t CB_PRE = 7;     // fp32 4 pages, pre_s in column 0 (rows = tokens)
constexpr uint32_t CB_H = 8;       // fp32 CPC pages, collapsed h tiles
constexpr uint32_t CB_S2 = 9;      // fp32 1 page, sum_c h^2 partial
constexpr uint32_t CB_INV = 10;    // fp32 1 page, 1/rms(h) in column 0
constexpr uint32_t CB_OUT = 11;    // bf16 (or fp32) CPC pages, hn tiles
constexpr uint32_t CB_H16 = 12;    // PF_FPU: alias of CB_H (same allocation), the S2 matmul operand
constexpr uint32_t CB_RED = 14;    // fp32 1 page, root: logits z [tokens x 24]
constexpr uint32_t CB_PLANE = 15;  // fp32 1 page, root: z in the plane layout
constexpr uint32_t CB_COEF = 16;   // fp32 1 page, root: pre | post | comb in the plane layout
constexpr uint32_t CB_SB = 17;     // fp32 3 pages, root: scale row, bias row, identity (PF_FPU diag mask) (resident)
constexpr uint32_t CB_INVR = 18;   // fp32 1 page, root: 1/rms(h) column
constexpr uint32_t CB_SCR = 19;    // fp32 2 pages, root: post / comb output tile assembly

// semaphores
constexpr uint32_t SEM_G1 = 0;  // root: round-1 partials arrived (P per row block)
constexpr uint32_t SEM_G2 = 1;  // root: round-2 partials arrived
constexpr uint32_t SEM_M1 = 2;  // member: round-1 message (pre) arrived
constexpr uint32_t SEM_M2 = 3;  // member: round-2 message (1/rms(h)) arrived

}  // namespace mhcpf
