// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

void kernel_main() {
    constexpr uint32_t shard_cb = get_compile_time_arg_val(0);

    const uint32_t input_shard_addr  = get_arg_val<uint32_t>(0);
    const uint32_t num_output_pages = get_arg_val<uint32_t>(1);
    const uint32_t num_ranges = get_arg_val<uint32_t>(2);
    uint32_t arg_index = 3;

    cb_reserve_back(shard_cb, num_output_pages);
    uint32_t l1_write_addr = get_write_ptr(shard_cb);
    for(uint32_t range_id = 0; range_id <num_ranges; range_id++) {
        uint32_t core_id_x = get_arg_val<uint32_t>(arg_index++);
        uint32_t core_id_y = get_arg_val<uint32_t>(arg_index++);
        uint32_t offset = get_arg_val<uint32_t>(arg_index++);
        uint32_t size = get_arg_val<uint32_t>(arg_index++);
        uint64_t noc_address = get_noc_addr(core_id_x, core_id_y,
                                        input_shard_addr + offset);
        noc_async_read(noc_address, l1_write_addr, size);
        l1_write_addr+=size;

    }
    noc_async_read_barrier();
    cb_push_back(shard_cb, num_output_pages);

}
