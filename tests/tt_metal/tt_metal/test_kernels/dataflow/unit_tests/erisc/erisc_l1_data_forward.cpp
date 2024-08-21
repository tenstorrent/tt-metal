// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

enum RX_MODE {
    FWD_L1 = 0,
    RX_ETH = 1,
    RX_TENSIX = 2
};

enum TX_MODE {
    PARK_L1 = 0,
    TX_ETH = 1,
    TX_TENSIX = 2
};

void kernel_main() {
    std::uint32_t num_bytes = get_arg_val<uint32_t>(0);
    RX_MODE rx_mode = get_arg_val<uint32_t>(1);
    TX_MODE tx_mode = get_arg_val<uint32_t>(2);

    std::uint32_t bytes_sent = 0;

    for (std::uint32_t i = 0; i < num_bytes; i += 64) {
        switch  (rx_mode) {
            case RX_MODE::RX_ETH:
                noc_semaphore_inc(sender_semaphore_noc_addr, 1);
                eth_noc_semaphore_wait(receiver_semaphore_addr_ptr, 1);
                noc_semaphore_set(receiver_semaphore_addr_ptr, 0);(64);
                break;
            case RX_MODE::RX_TENSIX:
                tensix_wait_for_bytes(64);
                break;
            case RX_MODE::FWD_L1:
                break;
        }

        if (tx_mode == TX_MODE::TX_ETH) {
            eth_send_bytes(src_addr, dst_addr, send_size);
            eth_wait_for_bytes(num_bytes); // Waits for write to complete
        } else if (tx_mode == TX_MODE::TX_TENSIX) {
            tensix_send_bytes(0, 64);
        }

        bytes_sent += 64;
    }

    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);


    eth_wait_for_bytes(num_bytes);
    noc_semaphore_inc(sender_semaphore_noc_addr, 1);

    eth_noc_semaphore_wait(receiver_semaphore_addr_ptr, 1);
    noc_semaphore_set(receiver_semaphore_addr_ptr, 0);

    eth_send_bytes(
        local_eth_l1_src_addr, remote_eth_l1_dst_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
    eth_wait_for_receiver_done();
}
