// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t dQ_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dK_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t dV_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t sb = get_arg_val<uint32_t>(runtime_args_counter++);              // s-tile-row in current batch
    uint32_t dQK_block_base = get_arg_val<uint32_t>(runtime_args_counter++);  // head 0 of (b, sb), w=0
    uint32_t dV_block_base = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_dq = tt::CBIndex::c_0;
    constexpr uint32_t cb_dknope = tt::CBIndex::c_1;
    constexpr uint32_t cb_dv = tt::CBIndex::c_2;
    constexpr uint32_t cb_dkpe_in = tt::CBIndex::c_3;

    constexpr uint32_t Th = get_compile_time_arg_val(0);
    constexpr uint32_t Tn = get_compile_time_arg_val(1);
    constexpr uint32_t Tr = get_compile_time_arg_val(2);
    constexpr uint32_t Tv = get_compile_time_arg_val(3);
    constexpr uint32_t n_heads = get_compile_time_arg_val(4);
    constexpr uint32_t kq_HtWt = get_compile_time_arg_val(5);  // Ts * Th
    constexpr uint32_t v_HtWt = get_compile_time_arg_val(6);   // Ts * Tv
    constexpr uint32_t Ts = get_compile_time_arg_val(7);
    constexpr uint32_t block_size = get_compile_time_arg_val(8);

    constexpr auto dQ_args = TensorAccessorArgs<9>();
    constexpr auto dK_args = TensorAccessorArgs<dQ_args.next_compile_time_args_offset()>();
    constexpr auto dV_args = TensorAccessorArgs<dK_args.next_compile_time_args_offset()>();

    const auto dQ_addr_gen = TensorAccessor(dQ_args, dQ_addr);
    const auto dK_addr_gen = TensorAccessor(dK_args, dK_addr);
    const auto dV_addr_gen = TensorAccessor(dV_args, dV_addr);

    const uint32_t tile_bytes = get_tile_size(cb_dkpe_in);

    // End-of-batch jump to advance dQK_block_base across batch boundary:
    //   from b * H*kq_HtWt + (Ts-1)*Th  →  (b+1) * H*kq_HtWt
    //   = current + ((H-1)*Ts + 1) * Th
    constexpr uint32_t end_of_batch_jump_q = ((n_heads - 1U) * Ts + 1U) * Th;
    constexpr uint32_t end_of_batch_jump_v = ((n_heads - 1U) * Ts + 1U) * Tv;

    // dQ / dK_nope / dV stream in block_size chunks via the shared util; dK_pe (Tr tiles) is read as one
    // burst into cb_dkpe_in because the compute kernel sums all Tr tiles of a head at once.
    for (uint32_t block = 0U; block < num_blocks; ++block) {
        for (uint32_t h = 0U; h < n_heads; ++h) {
            const uint32_t head_dqk = dQK_block_base + h * kq_HtWt;  // dQ and dK share head_dim
            const uint32_t head_dv = dV_block_base + h * v_HtWt;

            read_full_row_tiles(cb_dq, dQ_addr_gen, Th, block_size, tile_bytes, head_dqk);
            read_full_row_tiles(cb_dknope, dK_addr_gen, Tn, block_size, tile_bytes, head_dqk);
            read_tiles_by_row(cb_dkpe_in, dK_addr_gen, head_dqk + Tn, Tr, tile_bytes, Tr);
            read_full_row_tiles(cb_dv, dV_addr_gen, Tv, block_size, tile_bytes, head_dv);
        }

        ++sb;
        if (sb < Ts) {
            dQK_block_base += Th;
            dV_block_base += Tv;
        } else {
            sb = 0U;
            dQK_block_base += end_of_batch_jump_q;
            dV_block_base += end_of_batch_jump_v;
        }
    }
}
