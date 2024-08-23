// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

// export TT_METAL_DPRINT_CORES='(0,0)-(0,3)' in order to see DPRINT messages

void kernel_main() {
    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t stick_size               = get_arg_val<uint32_t>(1);
    const uint32_t shard_height             = get_arg_val<uint32_t>(2);
    const uint32_t shard_width_bytes        = get_arg_val<uint32_t>(3);
    const uint32_t padded_offset_bytes      = get_arg_val<uint32_t>(4);
    const uint32_t start_id                 = get_arg_val<uint32_t>(5);
    const uint32_t current_core             = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;
    const InterleavedAddrGen<src_is_dram> s0 = {
        .bank_base_address = src_addr,
        .page_size = stick_size
    };
    uint32_t stick_id = start_id;
    cb_reserve_back(cb_id_in0, shard_height);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    DPRINT_DATA0(DPRINT << "Core (0," << current_core << "): ");
    for (uint32_t h = 0; h < shard_height; ++h) {
        uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
        noc_async_read(src_noc_addr, l1_write_addr, stick_size);
        // print both BFloat16 values that are packed into the page
        uint32_t* read_ptr = (uint32_t*)l1_write_addr;
        DPRINT_DATA0(DPRINT << (uint16_t)*read_ptr << " ");
        DPRINT_DATA0(DPRINT << (uint16_t)(*read_ptr >> 16) << " ");
        stick_id++;
        l1_write_addr += padded_offset_bytes;
    }
    DPRINT_DATA0(DPRINT << ENDL());
    noc_async_read_barrier();
    cb_push_back(cb_id_in0, shard_height);
}
