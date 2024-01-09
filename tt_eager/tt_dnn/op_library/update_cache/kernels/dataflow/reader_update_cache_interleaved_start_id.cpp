// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

//#include "debug/dprint.h"

void kernel_main() {
    uint32_t cache_addr  = get_arg_val<uint32_t>(0);
    uint32_t input_addr  = get_arg_val<uint32_t>(1);
    uint32_t Wt          = get_arg_val<uint32_t>(2);
    uint32_t B           = get_arg_val<uint32_t>(3);
    uint32_t cache_HtWt  = get_arg_val<uint32_t>(4);
    uint32_t cache_start_id = get_arg_val<uint32_t>(5);
    uint32_t input_start_id = get_arg_val<uint32_t>(6);

    constexpr bool cache_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool input_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t cache_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t input_cb_id = get_compile_time_arg_val(3);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t cache_tile_bytes = get_tile_size(cache_cb_id);
    const DataFormat cache_data_format = get_dataformat(cache_cb_id);
    const uint32_t input_tile_bytes = get_tile_size(input_cb_id);
    const DataFormat input_data_format = get_dataformat(input_cb_id);

    const InterleavedAddrGenFast<cache_is_dram> s0 = {
        .bank_base_address = cache_addr,
        .page_size = cache_tile_bytes,
        .data_format = cache_data_format
    };
    const InterleavedAddrGenFast<input_is_dram> s1 = {
        .bank_base_address = input_addr,
        .page_size = input_tile_bytes,
        .data_format = input_data_format
    };

    uint32_t input_id = input_start_id;
    uint32_t cache_id = cache_start_id;

    for (uint32_t b = 0; b < B / 32; b++) {
        cb_reserve_back(input_cb_id, Wt);
        uint32_t input_l1_write_addr = get_write_ptr(input_cb_id);
        for (uint32_t i = 0; i < Wt; i++) {
            noc_async_read_tile(input_id, s1, input_l1_write_addr);
            input_l1_write_addr += input_tile_bytes;
            input_id++;
        }
        noc_async_read_barrier();
        cb_push_back(input_cb_id, Wt);
        for (uint32_t u = 0; u < 32; u++) {
            cb_reserve_back(cache_cb_id, Wt);
            uint32_t cache_l1_write_addr = get_write_ptr(cache_cb_id);
            for (uint32_t i = 0; i < Wt; i++) {
                noc_async_read_tile(cache_id, s0, cache_l1_write_addr);
                cache_l1_write_addr += cache_tile_bytes;
                cache_id++;
            }
            noc_async_read_barrier();
            cb_push_back(cache_cb_id, Wt);
            cache_id = cache_id - Wt + cache_HtWt;
        }
    }
}
