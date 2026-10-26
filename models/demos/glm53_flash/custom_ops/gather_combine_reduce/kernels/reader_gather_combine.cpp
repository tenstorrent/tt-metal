// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader of the fused gather + combine-reduce (tt-blaze G53F-0190). Replaces the post_combine_reduce reader
// (deepseek_prefill/post_combine_reduce/device/kernels/deepseek_moe_post_combine_reduce_reader.cpp) and the
// ttnn.embedding gather in front of it: instead of reading comb[t, j] (a gathered copy of table[pos[t, j]]) it reads
// table[pos[t, j]] directly, and only the rows the compute kernel uses. The compute kernel (unchanged) always processes
// a token's pick 0 and skips every later pick whose routing weight is zero; in the regions MoE the weight is zero
// exactly at the non-local picks, whose slot is the dummy slot, so a pick j > 0 at the dummy slot is not read (its CB
// rows are pushed unfilled and popped unread). The rows the compute does read are byte-identical to comb's, so the
// result is bit-identical. A token's (up to k) row reads are in flight together, one barrier per token.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"

constexpr uint32_t cb_combine_input_id = tt::CBIndex::c_0;
constexpr uint32_t num_experts = get_compile_time_arg_val(0);
constexpr uint32_t emb_dim_cb_tiles = get_compile_time_arg_val(1);
constexpr uint32_t emb_dim_bytes = get_compile_time_arg_val(2);
constexpr uint32_t dummy_slot = get_compile_time_arg_val(3);
constexpr uint32_t cb_pos_id = get_compile_time_arg_val(4);
constexpr uint32_t pos_page_bytes = get_compile_time_arg_val(5);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
constexpr auto table_args = TensorAccessorArgs<7>();
constexpr auto pos_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
constexpr uint32_t TOKENS_PER_CHUNK = 32;
constexpr uint32_t chunk_pos_bytes = TOKENS_PER_CHUNK * num_experts * 4;

void kernel_main() {
    const uint32_t table_addr = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_arg_val<uint32_t>(1);
    uint32_t token_start_idx = get_arg_val<uint32_t>(2);
    const uint32_t num_chunks = get_arg_val<uint32_t>(3);

    const auto s_table = TensorAccessor(table_args, table_addr, emb_dim_bytes);
    const auto s_pos = TensorAccessor(pos_args, pos_addr, pos_page_bytes);

    Noc noc;
    CircularBuffer cb_in(cb_combine_input_id), cb_pos(cb_pos_id);
    cb_pos.reserve_back(1);  // private scratch: reserved once, never pushed
    const uint32_t pos_base = get_write_ptr(cb_pos_id);
    const CoreLocalMem<volatile uint32_t> pos(pos_base);

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        // the chunk's 32 x k slots: one contiguous read (token_start is a multiple of 32: 1 KB aligned at k = 8)
        noc.async_read(
            s_pos,
            CoreLocalMem<uint32_t>(pos_base),
            chunk_pos_bytes,
            {.page_id = 0, .offset_bytes = token_start_idx * num_experts * 4u},
            {});
        noc.async_read_barrier();
        for (uint32_t t = 0; t < TOKENS_PER_CHUNK; ++t) {
            cb_in.reserve_back(num_experts * emb_dim_cb_tiles);
            for (uint32_t j = 0; j < num_experts; ++j) {
                const uint32_t slot = pos[t * num_experts + j];
                if (j == 0 || slot != dummy_slot) {
                    noc.async_read(
                        s_table,
                        cb_in,
                        emb_dim_bytes,
                        {.page_id = slot},
                        {.offset_bytes = j * emb_dim_cb_tiles * tile_bytes});
                }
            }
            noc.async_read_barrier();
            cb_in.push_back(num_experts * emb_dim_cb_tiles);
        }
        token_start_idx += TOKENS_PER_CHUNK;
    }
}
