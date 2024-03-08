// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>

void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();

        // eth_wait_for_bytes(16);
        // eth_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_done();

        // eth_send_bytes(handshake_register_address,handshake_register_address, 16);
        // eth_wait_for_receiver_done();
    }
}

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);
    constexpr std::uint32_t total_num_message_sends = get_compile_time_arg_val(2);
    constexpr std::uint32_t NUM_TRANSACTION_BUFFERS = get_compile_time_arg_val(3);
    constexpr bool src_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t MAX_NUM_CHANNELS = NUM_TRANSACTION_BUFFERS;
    static_assert(MAX_NUM_CHANNELS % 2 == 0, "MAX_NUM_CHANNELS must be <= 8");
    constexpr uint32_t CHANNEL_MASK = (MAX_NUM_CHANNELS - 1);
    std::array<std::uint32_t, MAX_NUM_CHANNELS> channels_active;
    for (uint8_t i = 0; i < NUM_TRANSACTION_BUFFERS; i++) {
        channels_active[i] = 0;
    }

    eth_setup_handshake(remote_eth_l1_dst_addr, true);

    kernel_profiler::mark_time(10);
    uint32_t j = 0;
    for (uint32_t i = 0; i < total_num_message_sends; i++) {
        kernel_profiler::mark_time(20);
        if (channels_active[j] != 0) {
            kernel_profiler::mark_time(21);
            eth_wait_for_receiver_channel_done(j);
            channels_active[j] = 0;
        }
        kernel_profiler::mark_time(22);
        eth_send_bytes_over_channel(
            local_eth_l1_src_addr,
            remote_eth_l1_dst_addr,
            num_bytes_per_send,
            j,
            num_bytes_per_send,
            num_bytes_per_send_word_size);
        channels_active[j] = 1;
        kernel_profiler::mark_time(23);
        j = (j + 1) & CHANNEL_MASK;
    }

    for (uint32_t j = 0; j < MAX_NUM_CHANNELS; j++) {
        kernel_profiler::mark_time(24);
        if (channels_active[j] != 0) {
            eth_wait_for_receiver_channel_done(j);
        }
        kernel_profiler::mark_time(25);
    }
    kernel_profiler::mark_time(11);
}
