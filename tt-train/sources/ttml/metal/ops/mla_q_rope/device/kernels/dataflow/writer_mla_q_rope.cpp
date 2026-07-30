// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t q_out_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t sb = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t q_block_base = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_nope = tt::CBIndex::c_4;
    constexpr uint32_t cb_rope_out = tt::CBIndex::c_16;

    constexpr uint32_t Tn = get_compile_time_arg_val(0);
    constexpr uint32_t Tr = get_compile_time_arg_val(1);
    constexpr uint32_t n_heads = get_compile_time_arg_val(2);
    constexpr uint32_t Ts = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_head = get_compile_time_arg_val(4);
    constexpr uint32_t kNopeChunkTiles = get_compile_time_arg_val(5);
    constexpr uint32_t Th = Tn + Tr;

    constexpr auto q_out_args = TensorAccessorArgs<6>();
    const auto q_out_gen = TensorAccessor(q_out_args, q_out_addr);

    const uint32_t tile_bytes = get_tile_size(cb_nope);

    constexpr uint32_t end_of_batch_jump = ((n_heads - 1U) * Ts + 1U) * Th;

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        for (uint32_t h = 0U; h < n_heads; ++h) {
            const uint32_t head_q = q_block_base + h * tiles_per_head;
            write_full_row_tiles(cb_nope, q_out_gen, Tn, kNopeChunkTiles, tile_bytes, head_q);
            write_tiles_by_row(cb_rope_out, q_out_gen, head_q + Tn, Tr, tile_bytes, Tr);
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
