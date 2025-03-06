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

    uint32_t first_noc;
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        first_noc = 0;
    } else {
        first_noc = noc_index;
    }

    for (uint32_t i = 0; i < 2; i++) {
        uint32_t noc_to_use = (i % 2) == 0 ? first_noc : 1 - first_noc;
        uint64_t dst_noc_addr = get_noc_addr(dst_noc_x, dst_noc_y, dst_addr, noc_to_use);
        noc_inline_dw_write(dst_noc_addr, value_to_write, 0xF, noc_to_use);
        if constexpr (noc_mode != DM_DYNAMIC_NOC) {
            break;
        }
        dst_addr += L1_ALIGNMENT;
        value_to_write++;
    }

    noc_async_write_barrier(noc_index);
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        noc_async_write_barrier(1 - noc_index);
    }
}
