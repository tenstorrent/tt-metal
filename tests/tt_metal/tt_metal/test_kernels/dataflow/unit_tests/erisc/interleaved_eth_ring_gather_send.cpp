// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "eth_l1_direct_ring_gather_utils.h"

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(2);
    std::uint32_t num_transfers = get_arg_val<uint32_t>(3);
    std::uint32_t global_start_idx = get_arg_val<uint32_t>(4);
    std::uint32_t src_addr = get_arg_val<uint32_t>(5);
    std::uint32_t dst_addr = get_arg_val<uint32_t>(6);
    std::uint32_t num_pages = get_arg_val<uint32_t>(7);
    std::uint32_t page_size = get_arg_val<uint32_t>(8);
    std::uint32_t sem_addr = get_arg_val<uint32_t>(9);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t receiver_noc_y = get_compile_time_arg_val(3);

    constexpr bool src_is_dram = get_compile_time_arg_val(4) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(5) == 1;

    const InterleavedAddrGen<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = page_size};
    const InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr, .page_size = page_size};

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint64_t receiver_semaphore_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, sem_addr);
    uint64_t receiver_data_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, remote_eth_l1_dst_addr);

    (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_eth_l1_src_addr)) = global_start_idx;
    uint32_t local_eth_l1_curr_src_addr = local_eth_l1_src_addr + 32;
    for (uint32_t curr_idx = 0; curr_idx < num_pages; ++curr_idx) {
        uint64_t src_noc_addr = get_noc_addr(curr_idx, s);
        noc_async_read(src_noc_addr, local_eth_l1_curr_src_addr, page_size);
        local_eth_l1_curr_src_addr += page_size;
    }
    eth_noc_async_read_barrier();

    local_eth_l1_curr_src_addr = local_eth_l1_src_addr + 32;
    for (uint32_t curr_idx = global_start_idx; curr_idx < global_start_idx + num_pages; ++curr_idx) {
        uint64_t dst_noc_addr = get_noc_addr(curr_idx, d);
        noc_async_write(local_eth_l1_curr_src_addr, dst_noc_addr, page_size);
        local_eth_l1_curr_src_addr += page_size;
    }
    // TODO: This block can overlap with eth transfer
    eth_noc_async_write_barrier();

    for (uint32_t i = 0; i < num_transfers; ++i) {
        eth_send_bytes(
            local_eth_l1_src_addr, remote_eth_l1_dst_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);

        eth_wait_for_remote_receiver_done_and_get_local_receiver_data(
            sender_semaphore_addr_ptr,
            receiver_semaphore_noc_addr,
            receiver_data_noc_addr,
            local_eth_l1_src_addr,
            num_bytes
        );

    }
}
