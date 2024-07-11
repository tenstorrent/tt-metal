// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    const uint32_t cache_addr  = get_arg_val<uint32_t>(0);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(1);
    uint32_t cache_tile_offset_B = get_arg_val<uint32_t>(2);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(3);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_input_cb_id = get_compile_time_arg_val(4);
    constexpr bool use_index_tensor = get_compile_time_arg_val(5) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(6);
    constexpr uint32_t cache_batch_num_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t Wbytes = get_compile_time_arg_val(9);

    DPRINT << "WRITER:"  << "cache_is_dram: " << (uint32_t)cache_is_dram << ENDL();
    DPRINT << "WRITER:"  << "cache_cb_id: " << cache_cb_id << ENDL();
    DPRINT << "WRITER:"  << "untilized_cache_cb_id: " << untilized_cache_cb_id << ENDL();
    DPRINT << "WRITER:"  << "untilized_cache2_cb_id: " << untilized_cache2_cb_id << ENDL();
    DPRINT << "WRITER:"  << "untilized_input_cb_id: " << untilized_input_cb_id << ENDL();
    DPRINT << "WRITER:"  << "use_index_tensor: " << (uint32_t)use_index_tensor << ENDL();
    DPRINT << "WRITER:"  << "cb_index_id: " << cb_index_id << ENDL();
    DPRINT << "WRITER:"  << "cache_batch_num_tiles: " << cache_batch_num_tiles << ENDL();
    DPRINT << "WRITER:"  << "Wt: " << Wt << ENDL();
    DPRINT << "WRITER:"  << "Wbytes: " << Wbytes << ENDL();

    DPRINT << "WRITER:"  << "cache_addr: " << cache_addr << ENDL();
    DPRINT << "WRITER:"  << "cache_start_id: " << cache_start_id << ENDL();
    DPRINT << "WRITER " << "cache_tile_offset_B: " << cache_tile_offset_B << ENDL();


    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    constexpr uint32_t TILE_HEIGHT = 32;

    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr,
        .page_size = cache_tile_bytes,
        .data_format = cache_data_format
    };

    uint32_t cache_id = cache_start_id;

    if constexpr (use_index_tensor) {
        cb_wait_front(cb_index_id, 1);
        uint32_t index_cb_ptr = get_read_ptr(cb_index_id);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_ptr);
        for (uint32_t b = 0; b < 32; ++b) {
            uint32_t index = index_ptr[b];
            DPRINT << "b=" << b << " index=" << index << ENDL();
        }
        const uint32_t update_idx = index_ptr[my_batch_idx];
        const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
        cache_id = cache_start_id;

        cache_tile_offset_B = update_idx % TILE_HEIGHT * Wbytes;
    }

    cb_wait_front(untilized_input_cb_id, Wt);
    uint64_t input_l1_read_addr = get_noc_addr(get_read_ptr(untilized_input_cb_id));

    // Wait on compute to untilize a block. Update that block in L1.
    cb_wait_front(untilized_cache_cb_id, Wt);
    cb_reserve_back(untilized_cache2_cb_id, Wt);
    uint32_t cache_l1_write_addr = get_read_ptr(untilized_cache_cb_id) + cache_tile_offset_B;
    noc_async_read(input_l1_read_addr, cache_l1_write_addr, Wbytes);
    noc_async_read_barrier();
    cb_push_back(untilized_cache2_cb_id, Wt);
    cb_pop_front(untilized_cache_cb_id, Wt); // NEW

    // Wait on compute to tilize an updated block. Write that block to DRAM
    cb_wait_front(cache_cb_id, Wt);
    uint32_t out_l1_read_addr = get_read_ptr(cache_cb_id);
    for(uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
        noc_async_write_tile(curr_cache_id, s0, out_l1_read_addr);
        out_l1_read_addr += cache_tile_bytes;
    }

    noc_async_writes_flushed();
    cb_pop_front(cache_cb_id, Wt);

    cb_pop_front(untilized_input_cb_id, Wt);

    // Delay syncing the writes to maximize perf.
    noc_async_write_barrier();
}
