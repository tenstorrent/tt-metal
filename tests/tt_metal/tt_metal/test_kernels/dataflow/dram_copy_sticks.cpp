// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr  = get_arg_val<uint32_t>(1);
    std::uint32_t bank_id        = get_arg_val<uint32_t>(2);

    std::uint32_t num_sticks      = get_arg_val<uint32_t>(3);
    std::uint32_t stick_size      =  get_arg_val<uint32_t>(4);
    for(uint32_t i = 0; i < 1; i++) {
        for(uint32_t stick_id = 0; stick_id < num_sticks; stick_id++) {
            // DRAM NOC src address
            std::uint64_t dram_buffer_src_noc_addr = get_noc_addr_from_bank_id<true>(bank_id, dram_buffer_src_addr);
            noc_async_read(dram_buffer_src_noc_addr, l1_buffer_addr, stick_size);
            noc_async_read_barrier();
            l1_buffer_addr += stick_size;
        }
    }
}
