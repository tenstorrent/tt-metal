// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_dram_noc_x = get_arg_val<uint32_t>(1);
    uint32_t dst_dram_noc_y = get_arg_val<uint32_t>(2);

    uint64_t dst_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_addr);

    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;
    uint32_t ublock_size_bytes = get_tile_size(cb_id_out0);
    uint32_t l1_read_addr = get_read_ptr(cb_id_out0);

    cb_wait_front(cb_id_out0, 1);
    auto l1_addr = get_write_ptr(cb_id_out0);

    auto l1_ptr = reinterpret_cast<char *>(l1_addr);
    // for(k : 0 -> 1024)
    for (uint32_t k = 0; k < 32; k++) {
        for (uint32_t j = 0; j < 32; j++) {
            uint32_t cur = *(uint32_t *)l1_ptr;
            *(float *)l1_ptr = static_cast<float>(cur) / (1 << 31 - 1) / 2;
            // *(float *)l1_ptr = 1.1f;
            l1_ptr += 4;
        }
    }
    // DPRINT << F32(*(float *)l1_addr) << ENDL();

    noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_id_out0, 1);
}
