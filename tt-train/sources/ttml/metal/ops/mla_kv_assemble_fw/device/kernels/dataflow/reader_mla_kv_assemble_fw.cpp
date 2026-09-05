// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

void kernel_main() {
    uint32_t runtime_args_counter = 0U;
    const uint32_t kv_up_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t k_pe_addr = get_arg_val<uint32_t>(runtime_args_counter++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t kv_up_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);
    uint32_t k_pe_tile_id = get_arg_val<uint32_t>(runtime_args_counter++);

    constexpr uint32_t cb_knope = tt::CBIndex::c_1;
    constexpr uint32_t cb_v = tt::CBIndex::c_2;
    constexpr uint32_t cb_kpe = tt::CBIndex::c_3;

    constexpr uint32_t Tn = get_compile_time_arg_val(0);  // qk_nope_dim / TILE_W
    constexpr uint32_t Tv = get_compile_time_arg_val(1);  // v_dim       / TILE_W
    constexpr uint32_t Tr = get_compile_time_arg_val(2);  // qk_rope_dim / TILE_W (k_pe tiles/block)
    constexpr uint32_t n_heads = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);  // tiles streamed per chunk

    constexpr auto kv_up_args = TensorAccessorArgs<5>();
    constexpr auto kpe_args = TensorAccessorArgs<kv_up_args.next_compile_time_args_offset()>();

    const auto kv_up_addr_gen = TensorAccessor(kv_up_args, kv_up_addr);
    const auto kpe_addr_gen = TensorAccessor(kpe_args, k_pe_addr);

    const uint32_t tile_bytes = get_tile_size(cb_kpe);
    constexpr uint32_t kv_tiles_per_head = Tn + Tv;

    // Each stream is fetched in block_size chunks (read_full_row_tiles handles the tail). CBs are
    // double-buffered, so the reader stays a chunk ahead of the writer on the opposite NoC; L1 use is
    // bounded by block_size, not by the per-head tile counts.
    for (uint32_t block = 0U; block < num_blocks; ++block) {
        // k_pe: all Tr tiles resident for the whole block (writer broadcasts them to every head).
        read_tiles_by_row(cb_kpe, kpe_addr_gen, k_pe_tile_id, Tr, tile_bytes, Tr);
        k_pe_tile_id += Tr;

        for (uint32_t h = 0U; h < n_heads; ++h) {
            // kv_up head layout is [k_nope (Tn) | v (Tv)]; demux into two streams.
            read_full_row_tiles(cb_knope, kv_up_addr_gen, Tn, block_size, tile_bytes, kv_up_tile_id);
            read_full_row_tiles(cb_v, kv_up_addr_gen, Tv, block_size, tile_bytes, kv_up_tile_id + Tn);
            kv_up_tile_id += kv_tiles_per_head;
        }
    }
}
