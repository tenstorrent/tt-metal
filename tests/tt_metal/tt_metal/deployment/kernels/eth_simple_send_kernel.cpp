// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "timestamp.hpp"

void kernel_main() {
    constexpr uint32_t transfer_size = get_compile_time_arg_val(0);
    constexpr uint32_t transfer_count = get_compile_time_arg_val(1);
    constexpr uint32_t send_delta_time_addr = get_compile_time_arg_val(2);
    constexpr uint32_t send_l1_address = get_compile_time_arg_val(3);
    constexpr uint32_t recv_l1_address = get_compile_time_arg_val(4);

    uint64_t start = timestamp();
    for (uint32_t i = 0; i < transfer_count; i++) {
        eth_send_bytes(send_l1_address, recv_l1_address, transfer_size);
        eth_wait_for_receiver_done();
    }
    uint64_t delta = timestamp() - start;

    *(uint64_t*)send_delta_time_addr = delta;
}
