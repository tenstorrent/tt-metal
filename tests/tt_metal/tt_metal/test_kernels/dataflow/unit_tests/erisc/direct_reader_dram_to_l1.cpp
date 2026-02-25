// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    std::uint32_t dram_buffer_src_addr_base = get_arg_val<uint32_t>(0);
    std::uint32_t dram_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_size = get_arg_val<uint32_t>(2);
    std::uint32_t local_eth_l1_addr_base = get_arg_val<uint32_t>(3);

    experimental::CoreLocalMem<std::uint32_t> local_buffer(local_eth_l1_addr_base);
    experimental::Noc noc;
    noc.async_read(
        experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
        local_buffer,
        dram_buffer_size,
        {.bank_id = dram_bank_id, .addr = dram_buffer_src_addr_base},
        {});
    noc.async_read_barrier();
}
