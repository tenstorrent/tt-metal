// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// BGE-M3 fused QKV-matmul output scatter-writer.
//
// STATUS: NOT YET IMPLEMENTED. This file is a placeholder so the sweep
// file's `SCATTER_WRITER_KERNEL_REL_PATH` reference is resolvable.
//
// Once `test_baseline_qkv_heads_timing` confirms the baseline matches the
// latest perf log, this kernel will be implemented as a fork of the stock
// matmul writer:
//
//     ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/
//         writer_bmm_tile_layout.cpp
//
// — with a single change: instead of writing every output tile to a single
// TensorAccessor `out_tensor_addr`, route each tile by its N-column index
// to one of three accessors (q_tensor_addr, k_tensor_addr, v_tensor_addr).
//
// Per-output layout for BGE-M3 B1/S512 (num_heads=16, head_dim=64):
//     QKV-fused output:  [B, 1, S, 3 * 16 * 64] = [..., 3072]
//     N-tile range 0..31  → Q[B, 16, S, 64]   (16 heads × 2 head_dim_tiles)
//     N-tile range 32..63 → K[B, 16, S, 64]
//     N-tile range 64..95 → V[B, 16, S, 64]
//
// Tile remap within Q:
//     N-tile n (0..31) belongs to head (n >> 1), head_dim_tile (n & 1).
//     Q dest tile index = batch_idx * (16 * S_t * 2)
//                       + (n >> 1) * (S_t * 2)            // head stride
//                       + h_tile_within_batch * 2          // S stride
//                       + (n & 1);                         // head_dim_tile
//
// (And similarly for K = n - 32 and V = n - 64.)
//
// The stock matmul writer iterates over (b, sbh, sbw, h, w) and writes one
// tile per inner iter; we keep that loop verbatim and only replace the
// `noc.async_write(... out_tensor, page_id=out_tensor_tile_id)` call with
// a branch on the N-column index that selects Q/K/V and remaps the page id.
//
// CT args (placeholder layout, to be finalized when wiring the ProgramDescriptor):
//   0: q_tiles_per_row    = num_q_heads * head_dim_tiles            (= 32)
//   1: k_tiles_per_row    = num_kv_heads * head_dim_tiles           (= 32)
//   2: head_dim_tiles     = head_dim / TILE_W                       (= 2)
//   3: head_htwt          = S_t * head_dim_tiles                    (= S/32 * 2)
//
// Runtime args (extends stock writer by 3, new args appended):
//   ... stock writer args ...
//   N+0: q_tensor_addr
//   N+1: k_tensor_addr
//   N+2: v_tensor_addr

#error "writer_qkv_scatter.cpp: NOT YET IMPLEMENTED. See file header for plan."
