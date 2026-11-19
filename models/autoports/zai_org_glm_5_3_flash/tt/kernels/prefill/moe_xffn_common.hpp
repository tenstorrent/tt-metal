// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 XSL expert FFN, one expert per core at a time (tt/prefill_tp.py EpMoe._xffn, GLM53_PF_MOE_FFN=xffn): the
// routed experts of a TP8 XSL chip are NARROW (every expert, width I / TP = 256 = 8 tiles). moe_fused_swiglu splits
// each expert over the whole grid (K over rows, hidden over columns, a reduce-scatter + an H multicast per expert),
// which for ~2 M-tiles per expert is synchronisation-bound. Here a core runs one expert's M group (<= MG tile rows)
// alone, start to end (units = (expert, group), round robin):
//   gate|up  G U [M, 2 W] = X [M, H] @ Wgu [H, 2 W]   (K chunks of KC tiles, fp32 partials by packer L1 accumulation)
//   act      A [M, W]     = silu(min(G, 10)) * clamp(U, -10, 10)            (clamped_silu_glu_tile)
//   down     Y [M, H]     = A @ Wd [W, H]           (N chunks of NC tiles, pack_untilize -> ROW_MAJOR rows)
// X rows are gathered from the token rows (x_ext, ROW_MAJOR [C + 32, H]) through the plan's disp table and tilized in
// place; Y rows go straight to the ROW_MAJOR expert output (dispatch row order, out_rm [CAP, H]) that the rm combine
// gathers -- no dispatch copy, no untilize pass. Weights: the served XSL tensors (bfloat4_b, ND-sharded: a gate_up
// shard = the (gate_j, up_j) tile pair of one K row, contiguous in one bank) read in place through TensorAccessor; the
// per-expert base addresses come from a [E, 2] uint32 table.
//
// Per expert and M group (<= MG tile rows) the reader pushes one control page to compute (CB_CTL: word 0 = the
// group's tile rows, 0 = end) and one to the writer (CB_WCTL: tile rows, valid rows, first dispatch row).
#pragma once

#include <cstdint>

namespace xffn {

constexpr uint32_t CB_CTL = 0;   // reader -> compute: [mt rows of the group] (uint32), 0 = no more work
constexpr uint32_t CB_WGU = 1;   // gate|up weight chunk: KC x 2W tiles (bfp4)
constexpr uint32_t CB_XRM = 2;   // x rows of one tile row, one K chunk: 32 rows x KC tiles ROW_MAJOR (bf16)
constexpr uint32_t CB_XT = 3;    // tilized x chunk: KC tiles per tile row (bf16)
constexpr uint32_t CB_WD = 4;    // down weight chunk: W x NC tiles (bfp4)
constexpr uint32_t CB_WCTL = 5;  // reader -> writer: [mt, valid rows, first dispatch row]
constexpr uint32_t CB_TAB = 6;   // reader scratch: counts, region starts, weight address table, disp rows
constexpr uint32_t CB_WTAB = 7;  // writer scratch (XFFN_X_ON_NOC1): the group's disp rows
constexpr uint32_t CB_XRM1 = 8;  // XFFN_X_SPLIT: the odd K chunks' x rows (writer -> compute); CB_XRM = the even ones
constexpr uint32_t CB_GUP = 16;  // fp32 gate|up partials: MG x 2W tiles
constexpr uint32_t CB_ACT = 17;  // activation tiles A: MG x W (bf16)
constexpr uint32_t CB_YRM = 18;  // y rows of one tile row, one N chunk: 32 rows x NC tiles ROW_MAJOR (bf16)
// the gang form (XFFN_GANG, moe_xffn_gang_*.cpp): the in-gang y ring. NAMES, never literals in the kernels, so a
// composite (16's kernels/pmk/gen_phases.py + moe_common.hpp) maps them to its own indices / semaphore ids. Ring
// contract: CB_RIN sits at the SAME L1 address on every core of a gang (the predecessor writes into its own
// get_write_ptr(CB_RIN) offset on the successor), and so do the two semaphores (get_semaphore(id) is uniform).
constexpr uint32_t CB_RIN = 19;          // the predecessor's partial blocks (bf16 tiles, 2 MG slots of NC tiles)
constexpr uint32_t SEM_RING_VALID = 0;   // +1 per block the predecessor landed in CB_RIN (grows; >= waits)
constexpr uint32_t SEM_RING_CREDIT = 1;  // +1 per CB_RIN slot the successor freed (grows; >= waits)

constexpr uint32_t TILE_ROWS = 32;
constexpr uint32_t BF16_TILE = 2048;

}  // namespace xffn
