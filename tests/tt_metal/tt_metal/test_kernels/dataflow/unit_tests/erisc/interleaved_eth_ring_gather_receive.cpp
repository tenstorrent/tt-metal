// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(1);
    std::uint32_t num_transfers = get_arg_val<uint32_t>(2);
    std::uint32_t dst_addr = get_arg_val<uint32_t>(3);
    std::uint32_t num_pages = get_arg_val<uint32_t>(4);
    std::uint32_t page_size = get_arg_val<uint32_t>(5);
    std::uint32_t sem_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t sender_noc_x = get_compile_time_arg_val(0);
    constexpr uint32_t sender_noc_y = get_compile_time_arg_val(1);

    constexpr bool dst_is_dram = get_compile_time_arg_val(2) == 1;

    const InterleavedAddrGen<dst_is_dram> d = {.bank_base_address = dst_addr, .page_size = page_size};

    volatile tt_l1_ptr uint32_t* receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint64_t sender_semaphore_noc_addr = get_noc_addr(sender_noc_x, sender_noc_y, sem_addr);

    for (uint32_t i = 0; i < num_transfers; ++i) {
        eth_wait_for_bytes(num_bytes);
        noc_semaphore_inc(sender_semaphore_noc_addr, 1);
        uint32_t start_idx = (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_eth_l1_src_addr));
        uint32_t local_eth_l1_curr_src_addr = local_eth_l1_src_addr + 32;
        for (uint32_t curr_idx = start_idx; curr_idx < start_idx + num_pages; ++curr_idx) {
            uint64_t dst_noc_addr = get_noc_addr(curr_idx, d);
            noc_async_write(local_eth_l1_curr_src_addr, dst_noc_addr, page_size);
            local_eth_l1_curr_src_addr += page_size;
        }
        eth_noc_async_write_barrier();
        eth_noc_semaphore_wait(receiver_semaphore_addr_ptr, 1);
        noc_semaphore_set(receiver_semaphore_addr_ptr, 0);
        eth_receiver_done();
    }
}
