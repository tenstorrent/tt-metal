// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on NCRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t dram_buffer_dst_addr_base = get_arg_val<uint32_t>(0);
    std::uint32_t dram_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t l1_buffer_src_addr_base = get_arg_val<uint32_t>(2);
    std::uint32_t dram_buffer_size = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_dst_addr = dram_buffer_dst_addr_base;

    // DRAM NOC dst address
    std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, dram_buffer_dst_addr);

    noc_async_write(l1_buffer_src_addr_base, dram_buffer_dst_noc_addr, dram_buffer_size);
    noc_async_write_barrier();
}
