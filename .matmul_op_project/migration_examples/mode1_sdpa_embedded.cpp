// Migration Examples: Mode 1 (Low-Level Tile) -- tt-train SDPA
// Call sites: T10, T11, T12
//
// These call sites are in the tt-train SDPA forward and backward kernels.
// They use matmul_tiles as one step in a larger compute pipeline (attention
// computation with softmax, masking, scaling, and gradient computation).
// Mode 1 is required because the matmul is deeply embedded in custom control
// flow with interleaved non-matmul operations.
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/matmul_op.h"
#include "experimental/circular_buffer.h"

// ============================================================================
// T10: tt-train SDPA forward -- Diagonal Q x K^T
// Source: tt-train/.../sdpa_fw/.../sdpa_fw_compute_kernel.cpp (lines 101-106)
//
// ORIGINAL CODE:
//   reconfig_data_format(cb_query, cb_key);
//   mm_init_short(cb_query, cb_key, /* transpose */ 1);
//   tile_regs_acquire();
//   for (tile_idx < qWt) {
//       matmul_tiles(cb_query, cb_key, tile_idx, tile_idx, matmul_accum_reg);
//   }
//   // ... apply mask, scale ...
//   tile_regs_commit();
//   // ... softmax, pack ...
//
// This computes the diagonal of Q @ K^T: for each tile position, the Q tile
// at index j is multiplied by the K tile at index j, accumulating into a
// single DST register. This is a 1x1 output tile from a 1xWt @ Wtx1 matmul.
// ============================================================================
namespace t10_sdpa_fw_qk {

void sdpa_fw_qk_snippet(uint32_t cb_query, uint32_t cb_key, uint32_t qWt, uint32_t matmul_accum_reg) {
    // --- NEW: MatmulOp for Q @ K^T ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_query,
        .in1_cb_id = cb_key,
        .out_cb_id = 0,     // not used in Mode 1 (caller manages pack)
        .transpose = true,  // K is transposed
    };
    ckernel::TileMatmulOp mm(cfg);
    // --- END NEW ---

    cb_wait_front(cb_key, qWt);
    reconfig_data_format(cb_query, cb_key);

    // --- NEW: init_short + matmul replaces mm_init_short + matmul_tiles ---
    mm.init_short();
    tile_regs_acquire();
    for (uint32_t tile_idx = 0; tile_idx < qWt; tile_idx++) {
        mm.matmul(tile_idx, tile_idx, matmul_accum_reg);
    }
    // --- END NEW ---

    // UNCHANGED: Apply mask/scale in DST, then commit, softmax, pack
    // ...
    tile_regs_commit();
    // ...
}

}  // namespace t10_sdpa_fw_qk

// ============================================================================
// T11: tt-train SDPA forward -- QK x V blocked
// Source: tt-train/.../sdpa_fw/.../sdpa_compute_utils.hpp (lines 149-154)
//
// ORIGINAL CODE:
//   mm_init_short(cb_qk_result, cb_value, /* transpose */ 0);
//   for (tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
//       tile_regs_acquire();
//       for (block_idx < block_size) {
//           matmul_tiles(cb_qk_result, cb_value, 0, tile_idx + block_idx, block_idx);
//       }
//       tile_regs_commit(); tile_regs_wait();
//       for (block_idx < block_size) { pack_tile(block_idx, cb_cur_mm_out); }
//       tile_regs_release();
//   }
//
// This is a 1xWt matmul (softmax output multiplied by V), computed in blocks
// of block_size output tiles. Each block: acquire -> block_size matmul_tiles
// -> commit -> pack block_size tiles -> release.
// ============================================================================
namespace t11_sdpa_fw_qkv {

void matmul_qk_by_v(
    uint32_t Wt, uint32_t block_size, uint32_t cb_qk_result, uint32_t cb_value, uint32_t cb_cur_mm_out) {
    cb_wait_front(cb_qk_result, 1);
    cb_wait_front(cb_value, Wt);
    cb_reserve_back(cb_cur_mm_out, Wt);

    // --- NEW: MatmulOp for QK @ V ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_qk_result,
        .in1_cb_id = cb_value,
        .out_cb_id = cb_cur_mm_out,
    };
    ckernel::TileMatmulOp mm(cfg);
    mm.init_short();  // replaces mm_init_short(...)
    // --- END NEW ---

    pack_reconfig_data_format(cb_cur_mm_out);
    reconfig_data_format(cb_qk_result, cb_value);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {
        tile_regs_acquire();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            // --- NEW: mm.matmul replaces matmul_tiles ---
            mm.matmul(0, tile_idx + block_idx, block_idx);
            // --- END NEW ---
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_cur_mm_out);
        }
        tile_regs_release();
    }
    cb_push_back(cb_cur_mm_out, Wt);
}

}  // namespace t11_sdpa_fw_qkv

// ============================================================================
// T12: tt-train SDPA backward -- gradient computation via matmul
// Source: tt-train/.../sdpa_bw/.../sdpa_bw_compute_utils.hpp (line 269 and
// the update_grad_query function at lines 275-322)
//
// ORIGINAL CODE (update_grad_query):
//   for (tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
//       tile_regs_acquire();
//       mm_init_short_with_dt(cb_grad_scores, cb_key, cb_grad_query_accum, 0);
//       for (block_idx < block_size) {
//           matmul_tiles(cb_grad_scores, cb_key, 0, tile_idx + block_idx, block_idx);
//       }
//       tile_regs_commit(); tile_regs_wait();
//       for (block_idx < block_size) { pack_tile(block_idx, cb_grad_query_accum); }
//       tile_regs_release();
//   }
//
// This computes dQ = dS @ K (gradient of query = gradient of scores multiplied
// by keys). Uses L1 accumulation across sequence blocks. The matmul pattern is
// identical to T11 but with init_short_with_dt for data format reconfig.
// ============================================================================
namespace t12_sdpa_bw_grad {

void update_grad_query(
    const uint32_t cb_grad_scores,
    const uint32_t cb_key,
    const uint32_t cb_grad_query_accum,
    const uint32_t tiles_per_row,
    const uint32_t block_size,
    const bool do_accumulate = false) {
    cb_wait_front(cb_grad_scores, 1);
    pack_reconfig_data_format(cb_grad_query_accum);

    if (!do_accumulate) {
        cb_reserve_back(cb_grad_query_accum, tiles_per_row);
    } else {
        pack_reconfig_l1_acc(true);
    }

    // --- NEW: MatmulOp for dQ = dS @ K ---
    ckernel::MatmulOpConfig cfg{
        .in0_cb_id = cb_grad_scores,
        .in1_cb_id = cb_key,
        .out_cb_id = cb_grad_query_accum,
    };
    ckernel::TileMatmulOp mm(cfg);
    // --- END NEW ---

    for (uint32_t tile_idx = 0; tile_idx < tiles_per_row; tile_idx += block_size) {
        tile_regs_acquire();

        // --- NEW: init_short_with_dt + matmul replaces
        //          mm_init_short_with_dt + matmul_tiles ---
        mm.init_short_with_dt(cb_grad_query_accum);
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            mm.matmul(0, tile_idx + block_idx, block_idx);
        }
        // --- END NEW ---

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
            pack_tile(block_idx, cb_grad_query_accum);
        }
        tile_regs_release();
    }

    if (do_accumulate) {
        pack_reconfig_l1_acc(false);
        cb_pop_front(cb_grad_query_accum, tiles_per_row);
        cb_reserve_back(cb_grad_query_accum, tiles_per_row);
    }
    cb_push_back(cb_grad_query_accum, tiles_per_row);
    cb_pop_front(cb_grad_scores, 1);
}

}  // namespace t12_sdpa_bw_grad
