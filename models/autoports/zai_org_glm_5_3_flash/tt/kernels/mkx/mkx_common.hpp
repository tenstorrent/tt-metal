// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel routed experts on the DS bank readers (tt/mk_experts.py): shared constants and CB indices.
//
// Expert weights stay in fused_experts' contract (tt/fused_decoder.py load_fused_experts): per expert a bfloat4_b
// gate_up [H, 2I] (tile columns [g0 | u0 | g1 | u1 | ...]) and down [I, H], DRAM ND-sharded [rows, 64] round robin
// over the 8 banks: shard j (gate_up: gate tile j + up tile j; down: output tiles 2j, 2j + 1) lives in bank j % 8, its
// tiles row-major and contiguous. Reader (bank b, index i of R) owns the shards j = b + 8 s, s = i, i + R, ... < 8: it
// computes m tile j = swiglu(x @ gate_up shard j) for those j, all readers all-gather m (the down K), and the reader
// computes output tiles 2j, 2j + 1 = sum over the union's experts of w[e, row] * (m_e @ down shard j) — the math and
// rounding points of fused_experts (matmul_gate_up.cpp): fp32 DST, gate_up packed fp32, SwiGLU on the SFPU, m packed
// bfloat8_b, down packed fp32, x routing weight (bf16 tile) -> bf16, bf16 accumulation over experts in ascending id
// order; routing weights scale * s / (sum of the row's selected s + eps), soft-float, bf16 round-to-nearest-even.
#pragma once

#include <cstdint>

#ifndef MKX_CB0
#define MKX_CB0 45
#endif

namespace mkx {

constexpr uint32_t CB_X = MKX_CB0 + 0;     // bf16 x, all K tiles, resident (KT_GU pages)
constexpr uint32_t CB_RING = MKX_CB0 + 1;  // bfloat4_b weight blocks (KB rows x 2 tiles), streamed
constexpr uint32_t CB_M = MKX_CB0 + 2;     // bfloat8_b m of one expert, all I tiles (gathered), the down's in0
constexpr uint32_t CB_MM = MKX_CB0 + 3;    // fp32 staging, 2 tiles
constexpr uint32_t CB_MO = MKX_CB0 + 4;    // bfloat8_b m tiles of this reader's shards (to the gather)
constexpr uint32_t CB_RS = MKX_CB0 + 5;    // bf16 routing weight tile of the current expert (row b = w[e, b])
constexpr uint32_t CB_ACC = MKX_CB0 + 6;   // running sum (bf16; MKX_XTP: fp32), 2 tiles per owned shard (+ 2 spare)
constexpr uint32_t CB_WTMP = MKX_CB0 + 7;  // bf16 weighted down output of one shard, 2 tiles
constexpr uint32_t CB_Y = MKX_CB0 + 8;     // bf16 final output tiles, 2 per owned shard
constexpr uint32_t CB_LIST = MKX_CB0 + 9;  // one page: [n, ids[n]] uint32 then the weights bf16 [n][32]
constexpr uint32_t CB_RAW = MKX_CB0 + 10;  // scratch: the ids tile, the score tiles, the expert address table
constexpr uint32_t CB_Y2 = MKX_CB0 + 11;   // MKX_Y2_INIT: bf16 tiles of another phase's output (2 per owned shard) that
                                           // seed the running sums (routed + shared expert in one partial)
// MKX_XTP (expert tensor parallelism, megakernel_plan §10e): this program computes ONE slice of 8 m tiles of every
// selected expert: on bank b the gate_up shard of slice tile b, its K split over the bank's 3 readers; readers 1, 2 of
// the bank send their fp32 gate | up partial (CB_PS) into reader 0's CB_PART (2 item slots x 2 senders x 2 tiles,
// written in place at the same L1 address, MKX_XTP_SEM counts them); reader 0 (the leader) adds them in K order,
// applies SwiGLU and publishes the slice's m tile b; the down then runs over K = the slice's 8 m tiles.
// MKX_XSL (batched expert TP): slice c (width W = MKX_XSL_W = 64 / TP m tiles) of every selected expert, the union
// taken GS = 64 / W experts at a time, each group ONE EP-shaped pass: group shard j is slice tile j % W of the group's
// expert j / W (gate_up shard W c + j % W, in bank j % 8, full K, read by EP's owner of shard j), m = SwiGLU x the
// expert's routing weight (CB_RS, one tile per owned shard with an expert), the 64-tile m all-gather as EP, then per
// owned output shard the down over K = W slice rows x the group's experts (fp32 running sums over the groups). No
// partials, one gather per group (TP8: 8 experts, TP2: 2).
constexpr uint32_t CB_PART = MKX_CB0 + 12;  // fp32, 8 pages (leader: the senders' partials, slot = pos & 1)
constexpr uint32_t CB_PS = MKX_CB0 + 13;    // fp32, 2 pages (sender: this reader's partial, sent by the NCRISC)
// MKX_XINV (the sparse layer's deferred FFN 1/rms(h), mk_sparse.py GLM53_MK_SPARSE_DEFER_INV): x = hn2 arrives as h * w
// without 1/rms(h); the per-row 1/rms(h) tile (column 0, fp32) scales gate | up before SwiGLU (the matmul is linear in
// x)
constexpr uint32_t CB_XINV = MKX_CB0 + 14;  // fp32, 1 page

#if defined(MKX_XTP)
constexpr uint32_t XTP_PART_SLOT_TILES = 4;  // one item slot of CB_PART: 2 senders x (gate, up)
#endif

constexpr uint32_t BLOCK_TILES_COLS = 2;  // a shard is 2 tile columns wide
constexpr uint32_t LIST_IDS = 2;          // word offset of ids in the list page (word 0 = n, word 1 = reserved)
constexpr uint32_t MAX_EXPERTS = 288;
constexpr uint32_t LIST_BYTES = 20480;  // the list page (mk_experts.py LIST_BYTES)
constexpr uint32_t RAW_BYTES = 26624;   // the CB_RAW scratch page (mk_experts.py RAW_BYTES)
// MKX_LIVE (the stack's position page, int32 [1, 32]: row b < 0 = an idle decode row): 128 B at the end of CB_RAW,
// above xw's routing work arrays (static_assert in xw.cpp)
constexpr uint32_t LIVE_OFF = RAW_BYTES - 128;
// MKX_RW_XA: xw leaves [its selection / rank scratch, the weights tile] addresses in the last words of the list page
// (behind the largest weight-row region: 1152 + 18944 B), xa scatters the weights from there
constexpr uint32_t LIST_XA_WORD = LIST_BYTES / 4 - 4;
constexpr uint32_t RW_TOPK_XA = 8;  // top-8 (xw's TOPK, static_assert there)

// element (row, col) of a 32x32 tile in face layout, in elements
inline uint32_t tile_elem(uint32_t row, uint32_t col) {
    return (((row >> 4) << 1) + (col >> 4)) * 256 + (row & 15u) * 16 + (col & 15u);
}

}  // namespace mkx
