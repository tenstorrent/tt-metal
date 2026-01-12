// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"

void kernel_main() {
    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t dram_bank_id = get_arg_val<uint32_t>(1);
    std::uint32_t remaining_bytes = get_arg_val<uint32_t>(2);
    std::uint32_t num_loops = get_arg_val<uint32_t>(3);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(4);

    experimental::Noc noc;
    experimental::AllocatorBank<experimental::AllocatorBankType::DRAM> src_dram;
    experimental::CoreLocalMem<std::uint32_t> dst_l1(eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE);

    // DRAM NOC src address
    for (uint32_t i = 0; i < num_loops; i++) {
        noc.async_read(src_dram, dst_l1, num_bytes, {.bank_id = dram_bank_id, .addr = dram_buffer_src_addr}, {});
        noc.async_read_barrier();

        eth_send_bytes(dst_l1.get_address(), dst_l1.get_address(), num_bytes);
        eth_wait_for_receiver_done();
        dram_buffer_src_addr += num_bytes;
    }

    noc.async_read(src_dram, dst_l1, remaining_bytes, {.bank_id = dram_bank_id, .addr = dram_buffer_src_addr}, {});
    noc.async_read_barrier();

    eth_send_bytes(dst_l1.get_address(), dst_l1.get_address(), remaining_bytes);
    eth_wait_for_receiver_done();
}
