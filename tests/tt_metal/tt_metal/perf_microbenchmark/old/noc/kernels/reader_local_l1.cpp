// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);
    uint32_t l1_buffer_noc_x = get_arg_val<uint32_t>(1);
    uint32_t l1_buffer_noc_y = get_arg_val<uint32_t>(2);
    uint32_t iter_cnt = get_arg_val<uint32_t>(3);
    uint32_t cb_tile_cnt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id);
    uint32_t cb_addr = get_write_ptr(cb_id);
    cb_reserve_back(cb_id, cb_tile_cnt);

    {
        DeviceZoneScopedN("NOC-FOR-LOOP");
        for (uint32_t i = 0; i < iter_cnt; i++) {
            uint32_t tmp_cb_addr = cb_addr;
            for (uint32_t j = 0; j < cb_tile_cnt; ++j) {
                uint64_t l1_buffer_noc_addr = get_noc_addr(l1_buffer_noc_x, l1_buffer_noc_y, l1_buffer_addr);
                noc_async_read(l1_buffer_noc_addr, tmp_cb_addr, single_tile_size_bytes);
                l1_buffer_addr += single_tile_size_bytes;
                tmp_cb_addr += single_tile_size_bytes;
            }
            noc_async_read_barrier();
        }
    }
}
