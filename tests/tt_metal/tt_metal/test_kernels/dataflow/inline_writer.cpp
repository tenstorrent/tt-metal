// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/debug/dprint.h"
#include "experimental/endpoints.h"

void kernel_main() {
    uint32_t dst_noc_x = get_arg_val<uint32_t>(0);
    uint32_t dst_noc_y = get_arg_val<uint32_t>(1);
    uint32_t dst_addr = get_arg_val<uint32_t>(2);
    uint32_t value_to_write = get_arg_val<uint32_t>(3);
    uint32_t num_writes = get_arg_val<uint32_t>(4);
    uint32_t dst_addr_increment = get_arg_val<uint32_t>(5);

    experimental::Noc noc(noc_index);
    experimental::Noc other_noc(1 - noc_index);

    experimental::Noc* noc_to_use = nullptr;
    for (uint32_t i = 0; i < num_writes; i++) {
        if constexpr (noc_mode == DM_DYNAMIC_NOC) {
            noc_to_use = (i % 2) == 0 ? &noc : &other_noc;
        } else {
            noc_to_use = &noc;
        }

        noc_to_use->inline_dw_write(
            experimental::UnicastEndpoint(),
            value_to_write,
            {.noc_x = dst_noc_x, .noc_y = dst_noc_y, .addr = dst_addr});
        dst_addr += dst_addr_increment;
        value_to_write++;
    }

    noc.async_write_barrier();
    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        other_noc.async_write_barrier();
    }
}
