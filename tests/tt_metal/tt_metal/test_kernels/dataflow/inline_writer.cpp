// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/dprint.h"

void kernel_main() {
    uint32_t dst_noc_x = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(1);
    uint32_t dst_addr = get_arg_val<uint32_t>(2);
    uint32_t value_to_write = get_arg_val<uint32_t>(3);
    uint32_t num_writes = get_arg_val<uint32_t>(4);
    uint32_t dst_addr_increment = get_arg_val<uint32_t>(5);

    for (uint32_t i = 0; i < num_writes; i++) {
        uint32_t noc_to_use;
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            noc_to_use = (i % 2) == 0 ? noc_index : 1 - noc_index;
        } else {
            noc_to_use = noc_index;
        }

        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr, noc_to_use);
        noc_inline_dw_write(dst_noc_addr, value_to_write, 0xF, noc_to_use);
        dst_addr += dst_addr_increment;
        value_to_write++;
    }

    noc_async_write_barrier(noc_index);
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        noc_async_write_barrier(1 - noc_index);
    }
}
