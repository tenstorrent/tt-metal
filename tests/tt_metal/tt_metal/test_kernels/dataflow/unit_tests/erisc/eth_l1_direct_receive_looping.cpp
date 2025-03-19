// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <algorithm>

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */

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
    std::size_t num_bytes_ = get_arg_val<uint32_t>(2);
    std::uint32_t num_loops_ = get_arg_val<uint32_t>(3);
    std::uint32_t num_sends_per_loop_ = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    constexpr std::uint32_t num_bytes = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_loops = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_sends_per_loop = get_compile_time_arg_val(4);

    // Handshake first before timestamping to make sure we aren't measuring any
    // dispatch/setup times for the kernels on both sides of the link.
    eth_setup_handshake(remote_eth_l1_dst_addr, false);

    uint32_t wrap_mask = num_sends_per_loop - 1;
    uint32_t j = 0;
    for (uint32_t i = 0; i < num_loops; i += num_sends_per_loop) {
        eth_wait_for_bytes(num_bytes * std::min<uint32_t>(num_loops - i, num_sends_per_loop));

        eth_receiver_done();

        j = (j + 1) & wrap_mask;
    }

    if (j != 0) {
        eth_receiver_done();
    }

    for (int i = 0; i < 3000000; i++);
}
