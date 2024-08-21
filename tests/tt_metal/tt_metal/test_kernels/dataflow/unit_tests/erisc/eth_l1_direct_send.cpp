// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::uint32_t num_bytes = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    eth_send_bytes(
        local_eth_l1_src_addr, remote_eth_l1_dst_addr, num_bytes, num_bytes_per_send, num_bytes_per_send_word_size);
    eth_wait_for_receiver_done();
}
