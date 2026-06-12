// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

// Compute kernel for the Q RoPE forward op. Applies rotary position embeddings to q_pe only.
//
// q_nope bypasses compute (reader -> writer). For every (batch, sequence-tile) block this kernel
// processes all n_heads heads on Tr rope tiles:
//   out_pe = (q_pe * cos) + (rotate(q_pe) * sin), where rotate(q_pe) is matmul with trans_mat.
//
// DST holds 4 tile slots with fp32_dest_acc_en, 8 without. Tr <= max_dst always, so batch up to
// (max_dst / Tr) heads per acquire to fill DST when Tr is small (e.g. Tr=1 -> 4 heads).

#if defined(FP32_DEST_ACC_EN)
constexpr uint32_t kMaxDstTiles = 4U;
#else
constexpr uint32_t kMaxDstTiles = 8U;
#endif

constexpr uint32_t q_pe_cb = get_compile_time_arg_val(0);
constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);
constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
constexpr uint32_t rope_out_cb = get_compile_time_arg_val(7);
constexpr uint32_t Tr = get_compile_time_arg_val(8);
constexpr uint32_t n_heads = get_compile_time_arg_val(9);
constexpr uint32_t kMaxChunkHeads = kMaxDstTiles / Tr;
constexpr uint32_t kMaxChunkTiles = kMaxChunkHeads * Tr;
constexpr uint32_t kTailHeads = n_heads % kMaxChunkHeads;
constexpr uint32_t kTailTiles = kTailHeads * Tr;

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    mm_init(q_pe_cb, trans_mat_cb, rotated_in_interm_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, rope_out_cb);

    cb_wait_front(trans_mat_cb, 1U);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        cb_wait_front(cos_cb, Tr);
        cb_wait_front(sin_cb, Tr);

        for (uint32_t head = 0U; head < n_heads; head += kMaxChunkHeads) {
            uint32_t chunk_tiles = kMaxChunkTiles;
            if (kTailHeads != 0U && head + kMaxChunkHeads > n_heads) {
                chunk_tiles = kTailTiles;
            }

            cb_wait_front(q_pe_cb, chunk_tiles);
            cb_reserve_back(rotated_in_interm_cb, chunk_tiles);
            cb_reserve_back(sin_interm_cb, chunk_tiles);
            cb_reserve_back(cos_interm_cb, chunk_tiles);
            cb_reserve_back(rope_out_cb, chunk_tiles);

            mm_init_short(q_pe_cb, trans_mat_cb);

            // rotate(q_pe) via matmul with trans_mat -> rotated_in_interm_cb.
            tile_regs_acquire();
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                matmul_tiles(q_pe_cb, trans_mat_cb, j, 0U, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(rotated_in_interm_cb);
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                pack_tile(j, rotated_in_interm_cb, j);
            }
            tile_regs_release();
            cb_push_back(rotated_in_interm_cb, chunk_tiles);
            cb_wait_front(rotated_in_interm_cb, chunk_tiles);

            // rotated_q_pe * sin -> sin_interm_cb.
            mul_tiles_init(rotated_in_interm_cb, sin_cb);
            tile_regs_acquire();
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                const uint32_t trig_idx = j % Tr;
                mul_tiles(rotated_in_interm_cb, sin_cb, j, trig_idx, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(sin_interm_cb);
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                pack_tile(j, sin_interm_cb, j);
            }
            tile_regs_release();
            cb_push_back(sin_interm_cb, chunk_tiles);
            cb_pop_front(rotated_in_interm_cb, chunk_tiles);

            // q_pe * cos -> cos_interm_cb.
            mul_tiles_init(q_pe_cb, cos_cb);
            tile_regs_acquire();
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                const uint32_t trig_idx = j % Tr;
                mul_tiles(q_pe_cb, cos_cb, j, trig_idx, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cos_interm_cb);
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                pack_tile(j, cos_interm_cb, j);
            }
            tile_regs_release();
            cb_push_back(cos_interm_cb, chunk_tiles);
            cb_pop_front(q_pe_cb, chunk_tiles);

            cb_wait_front(sin_interm_cb, chunk_tiles);
            cb_wait_front(cos_interm_cb, chunk_tiles);

            // cos_interm + sin_interm -> rope_out_cb.
            add_tiles_init(cos_interm_cb, sin_interm_cb);
            tile_regs_acquire();
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(rope_out_cb);
            for (uint32_t j = 0U; j < chunk_tiles; ++j) {
                pack_tile(j, rope_out_cb, j);
            }
            tile_regs_release();
            cb_push_back(rope_out_cb, chunk_tiles);
            cb_pop_front(sin_interm_cb, chunk_tiles);
            cb_pop_front(cos_interm_cb, chunk_tiles);
        }

        cb_pop_front(cos_cb, Tr);
        cb_pop_front(sin_cb, Tr);
    }

    cb_pop_front(trans_mat_cb, 1U);
}
