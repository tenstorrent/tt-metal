// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    const uint32_t cache_addr  = get_arg_val<uint32_t>(0);
    const uint32_t cache_start_id = get_arg_val<uint32_t>(1);
    const uint32_t index_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t my_batch_idx = get_arg_val<uint32_t>(3);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(2);
    constexpr bool use_index_tensor = get_compile_time_arg_val(3) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t cb_index_id = get_compile_time_arg_val(5);
    constexpr uint32_t cache_batch_num_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    const uint32_t log_base_2_of_page_size = get_compile_time_arg_val(8);
    const uint32_t index_stick_size_B = get_compile_time_arg_val(9);

    DPRINT << "READER: " << "cache_id_dram: " << (uint32_t)cache_is_dram << ENDL();
    DPRINT << "READER: " << "cache_cb_id: " << cache_cb_id << ENDL();
    DPRINT << "READER: " << "input_cb_id: " << input_cb_id << ENDL();
    DPRINT << "READER: " << "use_index_tensor: " << (uint32_t)use_index_tensor << ENDL();
    DPRINT << "READER: " << "index_is_dram: " << (uint32_t)index_is_dram << ENDL();
    DPRINT << "READER: " << "cb_index_id: " << cb_index_id << ENDL();
    DPRINT << "READER: " << "cache_batch_num_tiles: " << cache_batch_num_tiles << ENDL();
    DPRINT << "READER: " << "Wt: " << Wt << ENDL();
    DPRINT << "READER: " << "log_base_2_of_page_size: " << log_base_2_of_page_size << ENDL();
    DPRINT << "READER: " << "index_stick_size_B: " << index_stick_size_B << ENDL();

    DPRINT << "READER: " << "cache_addr: " << cache_addr << ENDL();
    DPRINT << "READER: " << "cache_start_id: " << cache_start_id << ENDL();
    DPRINT << "READER: " << "index_tensor_addr: " << index_tensor_addr << ENDL();

    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);

    constexpr uint32_t TILE_HEIGHT = 32;

    uint32_t cache_id = cache_start_id;

    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr,
        .page_size = cache_tile_bytes,
        .data_format = cache_data_format
    };

    if constexpr (use_index_tensor) {

        const InterleavedPow2AddrGen<index_is_dram> addrg = {
            .bank_base_address = index_tensor_addr,
            .log_base_2_of_page_size = log_base_2_of_page_size
        };

        cb_reserve_back(cb_index_id, 1);
        uint32_t index_cb_wr_ptr = get_write_ptr(cb_index_id);
        // index_tensor has one page to read
        uint64_t tensor_index_noc_addr = get_noc_addr(0, addrg);
        noc_async_read(tensor_index_noc_addr, index_cb_wr_ptr, index_stick_size_B);
        noc_async_read_barrier();
        cb_push_back(cb_index_id, 1);
        volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
        for (uint32_t b = 0; b < 32; ++b) {
            uint32_t index = index_ptr[b];
            DPRINT << "b=" << b << " index=" << index << ENDL();
        }
        const uint32_t update_idx = index_ptr[my_batch_idx];
        const uint32_t cache_batch_tile_offset = my_batch_idx * cache_batch_num_tiles;
        const uint32_t cache_start_id = cache_batch_tile_offset + (update_idx / TILE_HEIGHT) * Wt;
        cache_id = cache_start_id;
    }

    cb_reserve_back(input_cb_id, Wt);
    cb_push_back(input_cb_id, Wt);



    cb_reserve_back(cache_cb_id, Wt);
    uint32_t cache_l1_write_addr = get_write_ptr(cache_cb_id);
    for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
        noc_async_read_tile(curr_cache_id, s0, cache_l1_write_addr);
        cache_l1_write_addr += cache_tile_bytes;
    }

    noc_async_read_barrier();
    cb_push_back(cache_cb_id, Wt );
}
