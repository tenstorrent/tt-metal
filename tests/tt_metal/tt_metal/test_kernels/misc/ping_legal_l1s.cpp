// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    constexpr std::uint32_t base_l1_address = get_compile_time_arg_val(0);
    constexpr std::uint32_t intermediate_l1_addr = get_compile_time_arg_val(1);
    constexpr std::uint32_t size_bytes = get_compile_time_arg_val(2);

    for (uint32_t id = 0; id < NUM_L1_BANKS; id += 1) {
        uint32_t bank_id;
        bank_id = umodsi3_const_divisor<NUM_L1_BANKS>(id);
        uint32_t l1_address = base_l1_address + bank_to_l1_offset[bank_id];
        uint32_t noc_xy = l1_bank_to_noc_xy[noc_index][bank_id];
        uint64_t noc_addr = get_noc_addr_helper(noc_xy, l1_address);

        noc_async_read(noc_addr, intermediate_l1_addr, size_bytes);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* read_value_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(intermediate_l1_addr);
        *read_value_ptr = *read_value_ptr + 1;

        noc_async_write(intermediate_l1_addr, noc_addr, size_bytes);
        noc_async_write_barrier();
    }
}
