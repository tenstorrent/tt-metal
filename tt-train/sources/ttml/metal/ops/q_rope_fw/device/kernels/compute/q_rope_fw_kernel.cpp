// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

// q_nope bypasses compute (reader -> writer). Per block:
//   out_pe = (q_pe * cos) + (rotate(q_pe) * sin)
//
// Heads are grouped in steps of kDstBatchHeads. Each group batches matmul across
// (batch_heads * Tr) tiles to fill DST; sin/cos/add/rope run per head for writer sync.

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
constexpr uint32_t kDstBatchHeads = get_compile_time_arg_val(10);
constexpr uint32_t kTailHeads = n_heads % kDstBatchHeads;
constexpr uint32_t kFullBatchHeads = n_heads - kTailHeads;
constexpr uint32_t kFullBatchTiles = kDstBatchHeads * Tr;
constexpr uint32_t kTailBatchTiles = kTailHeads * Tr;

// Batched matmul (rotate q_pe) then per-head sin/cos/add/rope tail.
ALWI void process_rope_group(const uint32_t num_tiles) {
    cb_wait_front(q_pe_cb, num_tiles);
    cb_reserve_back(rotated_in_interm_cb, num_tiles);

    mm_init_short(q_pe_cb, trans_mat_cb);
    tile_regs_acquire();
    for (uint32_t j = 0U; j < num_tiles; ++j) {
        matmul_tiles(q_pe_cb, trans_mat_cb, j, 0U, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(rotated_in_interm_cb);
    for (uint32_t j = 0U; j < num_tiles; ++j) {
        pack_tile(j, rotated_in_interm_cb, j);
    }
    tile_regs_release();
    cb_push_back(rotated_in_interm_cb, num_tiles);
    cb_wait_front(rotated_in_interm_cb, num_tiles);

    for (uint32_t tile = 0U; tile < num_tiles; tile += Tr) {
        cb_reserve_back(sin_interm_cb, Tr);
        cb_reserve_back(cos_interm_cb, Tr);
        cb_reserve_back(rope_out_cb, Tr);

        mul_tiles_init(rotated_in_interm_cb, sin_cb);
        tile_regs_acquire();
        for (uint32_t j = 0U; j < Tr; ++j) {
            mul_tiles(rotated_in_interm_cb, sin_cb, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(sin_interm_cb);
        for (uint32_t j = 0U; j < Tr; ++j) {
            pack_tile(j, sin_interm_cb, j);
        }
        tile_regs_release();
        cb_push_back(sin_interm_cb, Tr);
        cb_pop_front(rotated_in_interm_cb, Tr);

        mul_tiles_init(q_pe_cb, cos_cb);
        tile_regs_acquire();
        for (uint32_t j = 0U; j < Tr; ++j) {
            mul_tiles(q_pe_cb, cos_cb, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cos_interm_cb);
        for (uint32_t j = 0U; j < Tr; ++j) {
            pack_tile(j, cos_interm_cb, j);
        }
        tile_regs_release();
        cb_push_back(cos_interm_cb, Tr);
        cb_pop_front(q_pe_cb, Tr);

        cb_wait_front(sin_interm_cb, Tr);
        cb_wait_front(cos_interm_cb, Tr);
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        tile_regs_acquire();
        for (uint32_t j = 0U; j < Tr; ++j) {
            add_tiles(cos_interm_cb, sin_interm_cb, j, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(rope_out_cb);
        for (uint32_t j = 0U; j < Tr; ++j) {
            pack_tile(j, rope_out_cb, j);
        }
        tile_regs_release();
        cb_push_back(rope_out_cb, Tr);
        cb_pop_front(sin_interm_cb, Tr);
        cb_pop_front(cos_interm_cb, Tr);
    }
}

void kernel_main() {
    const uint32_t num_blocks = get_arg_val<uint32_t>(0);

    mm_init(q_pe_cb, trans_mat_cb, rotated_in_interm_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, rope_out_cb);

    cb_wait_front(trans_mat_cb, 1U);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        cb_wait_front(cos_cb, Tr);
        cb_wait_front(sin_cb, Tr);

        for (uint32_t head_base = 0U; head_base < kFullBatchHeads; head_base += kDstBatchHeads) {
            process_rope_group(kFullBatchTiles);
        }
        if constexpr (kTailHeads != 0U) {
            process_rope_group(kTailBatchTiles);
        }

        cb_pop_front(cos_cb, Tr);
        cb_pop_front(sin_cb, Tr);
    }

    cb_pop_front(trans_mat_cb, 1U);
}
