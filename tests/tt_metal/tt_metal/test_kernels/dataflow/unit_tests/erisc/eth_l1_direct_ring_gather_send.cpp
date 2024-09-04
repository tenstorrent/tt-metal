// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "eth_l1_direct_ring_gather_utils.h"

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr_end = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(2);
    std::uint32_t num_transfers = get_arg_val<uint32_t>(3);
    std::uint32_t local_eth_l1_curr_src_addr = get_arg_val<uint32_t>(4);
    std::uint32_t sender_idx = get_arg_val<uint32_t>(5);
    std::uint32_t sem_addr = get_arg_val<uint32_t>(6);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_noc_x = get_compile_time_arg_val(2);
    constexpr uint32_t receiver_noc_y = get_compile_time_arg_val(3);

    volatile tt_l1_ptr uint32_t* sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);
    uint64_t receiver_semaphore_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, sem_addr);
    uint64_t receiver_data_noc_addr = get_noc_addr(receiver_noc_x, receiver_noc_y, remote_eth_l1_dst_addr);

    for (uint32_t i = 0; i < num_transfers; ++i) {
        eth_send_bytes(
            local_eth_l1_curr_src_addr, remote_eth_l1_dst_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
        if (sender_idx == 0) {
            sender_idx = num_transfers;
            local_eth_l1_curr_src_addr = local_eth_l1_src_addr_end;
        } else {
            local_eth_l1_curr_src_addr -= num_bytes;
            sender_idx--;
        }
        eth_wait_for_remote_receiver_done_and_get_local_receiver_data(
            sender_semaphore_addr_ptr,
            receiver_semaphore_noc_addr,
            receiver_data_noc_addr,
            local_eth_l1_curr_src_addr,
            num_bytes
        );

    }
}
