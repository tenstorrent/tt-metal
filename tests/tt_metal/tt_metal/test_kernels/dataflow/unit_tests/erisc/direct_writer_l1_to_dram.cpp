// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on NCRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */
void kernel_main() {
    std::uint32_t dram_buffer_dst_addr_base = get_arg_val<uint32_t>(0);
    std::uint32_t dram_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t dram_buffer_size = get_arg_val<uint32_t>(2);
    std::uint32_t local_eth_l1_addr_base = get_arg_val<uint32_t>(3);

    std::uint32_t dram_buffer_dst_addr = dram_buffer_dst_addr_base;

    // DRAM NOC dst address
    std::uint64_t dram_buffer_dst_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, dram_buffer_dst_addr);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dst_dram;
    experimental::CoreLocalMem<std::uint32_t> src_l1(local_eth_l1_addr_base);

    noc.async_write(src_l1, dst_dram, dram_buffer_size, {}, {.bank_id = dram_bank_id, .addr = dram_buffer_dst_addr});
    noc.async_write_barrier();
}
