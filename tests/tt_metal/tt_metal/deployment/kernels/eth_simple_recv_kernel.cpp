// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

void kernel_main() {
    constexpr uint32_t transfer_size = get_compile_time_arg_val(0);
    constexpr uint32_t transfer_count = get_compile_time_arg_val(1);

    for (uint32_t i = 0; i < transfer_count; i++) {
        eth_wait_for_bytes(transfer_size);
        eth_receiver_done();
    }
}
