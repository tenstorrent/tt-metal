// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t q_in_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t cos_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t sin_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t trans_mat_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    // This core's contiguous share of global work units (B * Ts); one block = one (batch, seq-tile) slice.
    const uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    // Sequence-tile row within the current batch [0, Ts); indexes cos/sin cache rows (no batch dim).
    uint32_t sb = get_arg_val<uint32_t>(runtime_args_counter++);
    // Tile id of head 0, width 0 for the current (batch, sb); head h uses q_block_base + h * tiles_per_head.
    uint32_t q_block_base = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_q_pe = tt::CBIndex::c_0;
    constexpr uint32_t cb_cos = tt::CBIndex::c_1;
    constexpr uint32_t cb_sin = tt::CBIndex::c_2;
    constexpr uint32_t cb_trans = tt::CBIndex::c_3;
    constexpr uint32_t cb_nope = tt::CBIndex::c_4;

    constexpr uint32_t Tn = get_compile_time_arg_val(0);
    constexpr uint32_t Tr = get_compile_time_arg_val(1);
    constexpr uint32_t n_heads = get_compile_time_arg_val(2);
    constexpr uint32_t Ts = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(4);
    constexpr uint32_t kNopeChunkTiles = get_compile_time_arg_val(5);
    constexpr uint32_t Th = Tn + Tr;

    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto cos_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto trans_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();

    const auto q_gen = TensorAccessor(q_args, q_in_addr);
    const auto cos_gen = TensorAccessor(cos_args, cos_addr);
    const auto sin_gen = TensorAccessor(sin_args, sin_addr);
    const auto trans_gen = TensorAccessor(trans_args, trans_mat_addr);

    const uint32_t tile_bytes = get_tile_size(cb_q_pe);

    // Advancing q_block_base after each block (all heads for one (batch b, sequence tile sb)):
    //   - Same batch, next sb:  q_block_base += Th          (one more 32-row slice on head 0's page)
    //   - Next batch, sb = 0:  q_block_base += end_of_batch_jump
    //
    // Tile id for head h, seq tile sb (width tile t = 0) is:
    //   (b * n_heads + h) * tiles_per_head + sb * Th
    // because each (b, h) owns a contiguous page of Ts*Th = tiles_per_head tiles.
    //
    // end_of_batch_jump: add to q_block_base when sb wraps (finished sb = Ts-1, next block is b+1, sb=0).
    // At the start of block (b, Ts-1), head 0 begins at:
    //   q_block_base = b * n_heads * tiles_per_head + (Ts - 1) * Th
    // After that block we need head 0 at (b+1, 0):
    //   (b + 1) * n_heads * tiles_per_head
    // So:
    //   end_of_batch_jump = (b+1)*H*tiles_per_head - [b*H*tiles_per_head + (Ts-1)*Th]
    //                     = H*tiles_per_head - (Ts-1)*Th
    //                     = H*Ts*Th - (Ts-1)*Th = ((H-1)*Ts + 1)*Th
    // Intuition: (H-1)*Ts*Th skips the other H-1 heads' pages in batch b; +Th steps past the
    // last seq tile of head 0 into the next batch's head-0 page.
    constexpr uint32_t end_of_batch_jump = ((n_heads - 1U) * Ts + 1U) * Th;

    // trans_mat is constant - load once per core.
    cb_reserve_back(cb_trans, 1U);
    const uint32_t trans_l1 = get_write_ptr(cb_trans);
    noc_async_read_tile(0, trans_gen, trans_l1);
    noc_async_read_barrier();
    cb_push_back(cb_trans, 1U);

    // cos/sin caches are [1, 1, S, qk_rope_dim] = Ts rows of Tr tiles with NO batch dim, so they are
    // indexed by sequence tile `sb` only (valid tile ids 0 .. Ts*Tr - 1) and every batch re-reads the
    // same rows. We must wrap sb at Ts: if it kept growing across a batch boundary, sb * Tr would point
    // past the cache and read out of bounds. Example (Ts = 4, Tr = 2, B = 2, all 8 blocks on one core):
    //   block: 0  1  2  3 | 4  5  6  7
    //   sb:    0  1  2  3 | 0  1  2  3   (wrapped, NOT 4 5 6 7)
    //   id:    0  2  4  6 | 0  2  4  6   (always < Ts*Tr = 8; without the wrap block 4 would read id 8)
    for (uint32_t block = 0U; block < num_blocks; ++block) {
        const uint32_t cos_sin_tile_id = sb * Tr;
        read_tiles_by_row(cb_cos, cos_gen, cos_sin_tile_id, Tr, tile_bytes, Tr);
        read_tiles_by_row(cb_sin, sin_gen, cos_sin_tile_id, Tr, tile_bytes, Tr);

        for (uint32_t h = 0U; h < n_heads; ++h) {
            const uint32_t head_q = q_block_base + h * tiles_per_head;
            read_full_row_tiles(cb_nope, q_gen, Tn, kNopeChunkTiles, tile_bytes, head_q);
            read_tiles_by_row(cb_q_pe, q_gen, head_q + Tn, Tr, tile_bytes, Tr);
        }

        ++sb;
        if (sb < Ts) {
            q_block_base += Th;
        } else {
            sb = 0U;
            q_block_base += end_of_batch_jump;
        }
    }
}
