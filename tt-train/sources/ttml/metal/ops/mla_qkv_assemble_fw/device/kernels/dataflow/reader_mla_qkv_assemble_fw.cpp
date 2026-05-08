// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    uint32_t q_pre_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t kv_up_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t k_pe_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t q_pre_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t kv_up_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t k_pe_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_q = tt::CBIndex::c_0;
    constexpr uint32_t cb_kv_up = tt::CBIndex::c_1;
    constexpr uint32_t cb_kpe = tt::CBIndex::c_2;

    constexpr uint32_t Th = get_compile_time_arg_val(0);                   // q tiles per head (Tn + Tr)
    constexpr uint32_t kv_tiles_per_head = get_compile_time_arg_val(1);    // Tn + Tv
    constexpr uint32_t kpe_tiles_per_block = get_compile_time_arg_val(2);  // Tr
    constexpr uint32_t n_heads = get_compile_time_arg_val(3);

    constexpr auto q_args = TensorAccessorArgs<4>();
    constexpr auto kv_up_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto kpe_args = TensorAccessorArgs<kv_up_args.next_compile_time_args_offset()>();

    const auto q_addr_gen = TensorAccessor(q_args, q_pre_addr);
    const auto kv_up_addr_gen = TensorAccessor(kv_up_args, kv_up_addr);
    const auto kpe_addr_gen = TensorAccessor(kpe_args, k_pe_addr);

    const uint32_t tile_bytes = get_tile_size(cb_kpe);

    for (uint32_t block = 0U; block < num_blocks; ++block) {
        // Block-level: load Tr k_pe tiles (peeked by writer for every head, popped at block end).
        read_tiles_by_row(cb_kpe, kpe_addr_gen, k_pe_tile_id, kpe_tiles_per_block, tile_bytes, kpe_tiles_per_block);
        k_pe_tile_id += kpe_tiles_per_block;

        // Per-head: stream Q tiles, then K-nope+V tiles. The writer drains them in the same order.
        for (uint32_t h = 0U; h < n_heads; ++h) {
            // Q for head h
            for (uint32_t w = 0U; w < Th; ++w) {
                cb_reserve_back(cb_q, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_q);
                noc_async_read_page(q_pre_tile_id + w, q_addr_gen, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_q, onetile);
            }
            q_pre_tile_id += Th;

            // KV (k_nope ∥ v) for head h
            for (uint32_t w = 0U; w < kv_tiles_per_head; ++w) {
                cb_reserve_back(cb_kv_up, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_kv_up);
                noc_async_read_page(kv_up_tile_id + w, kv_up_addr_gen, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_kv_up, onetile);
            }
            kv_up_tile_id += kv_tiles_per_head;
        }
    }
}
