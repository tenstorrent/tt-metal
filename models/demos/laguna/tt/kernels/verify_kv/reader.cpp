// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Verify KV write (Laguna; verify_kv_write.py), reader. One core per (set, row): set 0 writes K rows into the K cache,
// set 1 writes V rows into the V cache. Every core reads all row positions (cur_pos) and, for every row, the 64-byte
// page-table chunk holding that row's entry, works out each row's head-0 cache tile, and takes a role: the lowest row
// landing in a cache tile is that tile's leader (it reads the tile, merges its followers' rows in and writes it once),
// the other rows of the tile are followers (no cache read or write). The role goes to the writer through cb_meta:
// [is_leader, follower count, leader row, head-0 tile id, own row in the tile, (follower row, its row in the tile)...].
//
// Adapted from paged_cache's reader_paged_fused_update_cache_interleaved_start_id.cpp: the legacy cache-tile read
// loop (78608b96a50^, lines 157-172) and the group-mode role logic of jerrywangTT/laguna-s-merge-main (8437ef46690,
// GROUP_MODE block). Differences: the page-table row is not read whole (only the 64-byte chunks), and the head-0 tile
// id goes to the writer in cb_meta instead of the writer re-reading the page table.
//
// Compile-time args: cb_in_k, cb_in_v, cb_cache, cb_index, cb_meta, Wt, num_heads, block_size, block_size_t, St,
// num_rows, meta_bytes, then TensorAccessorArgs of the cache (K and V share one layout), cur_pos and page_table.
// Runtime args: set (0 = K, 1 = V), row. Common runtime args: K cache, V cache, cur_pos and page_table addresses.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_in_k = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in_v = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cache = get_compile_time_arg_val(2);
    constexpr uint32_t cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t num_heads = get_compile_time_arg_val(6);
    constexpr uint32_t block_size = get_compile_time_arg_val(7);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(8);
    constexpr uint32_t St = get_compile_time_arg_val(9);
    constexpr uint32_t num_rows = get_compile_time_arg_val(10);
    constexpr uint32_t meta_bytes = get_compile_time_arg_val(11);
    constexpr auto cache_args = TensorAccessorArgs<12>();
    constexpr auto index_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto page_table_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t head_offset_t = Wt * St;
    constexpr uint32_t SKIP = 0xFFFFFFFFu;

    const uint32_t set = get_arg_val<uint32_t>(0);
    const uint32_t my_row = get_arg_val<uint32_t>(1);

    const auto cache = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(set == 0 ? 0 : 1));
    const auto index = TensorAccessor(index_args, get_common_arg_val<uint32_t>(2));
    const auto page_table = TensorAccessor(page_table_args, get_common_arg_val<uint32_t>(3));

    // Kick off compute: the input buffer is this core's shard of the K (V) rows, already in L1
    const uint32_t cb_in = set == 0 ? cb_in_k : cb_in_v;
    cb_reserve_back(cb_in, Wt);
    cb_push_back(cb_in, Wt);

    // every row's position (one page; -1 = skip the row). cb_index is reader-only scratch (never pushed)
    cb_reserve_back(cb_index, 1);
    const uint32_t index_addr = get_write_ptr(cb_index);
    noc_async_read_page(0, index, index_addr);

    // the meta entry: role words, then one 64-byte page-table chunk per row
    cb_reserve_back(cb_meta, 1);
    const uint32_t meta_addr = get_write_ptr(cb_meta);
    const uint32_t chunks_addr = meta_addr + meta_bytes;
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta_addr);
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_addr);
    noc_async_read_barrier();

    for (uint32_t b = 0; b < num_rows; ++b) {
        const uint32_t idx = pos[b];
        if (idx == SKIP) {
            continue;
        }
        const uint32_t entry_offset = (idx / block_size) * sizeof(uint32_t);
        noc_async_read(page_table.get_noc_addr(b, entry_offset & ~63u), chunks_addr + b * 64, 64);
    }
    noc_async_read_barrier();

    uint32_t tile_of[num_rows];
    for (uint32_t b = 0; b < num_rows; ++b) {
        const uint32_t idx = pos[b];
        if (idx == SKIP) {
            tile_of[b] = SKIP;
            continue;
        }
        const uint32_t entry_addr = chunks_addr + b * 64 + (((idx / block_size) * sizeof(uint32_t)) & 63u);
        const uint32_t physical_block_id = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(entry_addr)[0];
        tile_of[b] = physical_block_id * num_heads * block_size_t * Wt + ((idx % block_size) / TILE_HEIGHT) * Wt;
    }

    const uint32_t mine = tile_of[my_row];
    uint32_t leader = SKIP;
    uint32_t followers = 0;
    if (mine != SKIP) {
        for (uint32_t b = 0; b < num_rows; ++b) {
            if (tile_of[b] != mine) {
                continue;
            }
            if (leader == SKIP) {
                leader = b;
            } else if (leader == my_row) {
                meta[5 + 2 * followers] = b;
                meta[6 + 2 * followers] = pos[b] % TILE_HEIGHT;
                ++followers;
            }
        }
    }
    const bool is_leader = leader == my_row;
    meta[0] = is_leader ? 1 : 0;
    meta[1] = followers;
    meta[2] = leader;
    meta[3] = mine == SKIP ? 0 : mine;
    meta[4] = mine == SKIP ? 0 : pos[my_row] % TILE_HEIGHT;
    cb_push_back(cb_meta, 1);

    // the cache tiles of every head (a follower or a skipped row pushes the buffer without reading: compute runs the
    // same untilize / tilize, the writer drops the result)
    uint32_t cache_id = mine;
    for (uint32_t h = 0; h < num_heads; ++h) {
        cb_reserve_back(cb_cache, Wt);
        if (is_leader) {
            uint32_t l1 = get_write_ptr(cb_cache);
            for (uint32_t t = 0; t < Wt; ++t) {
                noc_async_read_page(cache_id + t, cache, l1);
                l1 += cache.get_aligned_page_size();
            }
            noc_async_read_barrier();
        }
        cb_push_back(cb_cache, Wt);
        cache_id += head_offset_t;
    }
}
