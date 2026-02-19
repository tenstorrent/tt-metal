// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t remaining_bytes = get_arg_val<uint32_t>(2);
    std::uint32_t num_loops = get_arg_val<uint32_t>(3);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(4);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> dst_dram;
    experimental::CoreLocalMem<std::uint32_t> src_l1(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);

    // DRAM NOC dst address
    for (uint32_t i = 0; i < num_loops; i++) {
        eth_wait_for_bytes(num_bytes);

        noc.async_write(src_l1, dst_dram, num_bytes, {}, {.bank_id = dram_bank_id, .addr = dram_buffer_dst_addr});
        noc.async_write_barrier();

        eth_receiver_done();
        dram_buffer_dst_addr += num_bytes;
    }

    eth_wait_for_bytes(remaining_bytes);

    noc.async_write(src_l1, dst_dram, remaining_bytes, {}, {.bank_id = dram_bank_id, .addr = dram_buffer_dst_addr});
    noc.async_write_barrier();

    eth_receiver_done();
}
