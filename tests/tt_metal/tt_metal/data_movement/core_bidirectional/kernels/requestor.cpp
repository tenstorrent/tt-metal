// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

constexpr inline uint32_t increment_virtual_channel(uint32_t previous_virtual_channel) {
    // Increment the virtual channel, wrapping around if it exceeds the maximum
    return (previous_virtual_channel + 1) & 0b11;
}

// L1 to L1 send
void kernel_main() {
    // Compile-time arguments
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t l1_local_read_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(2);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(3);
    constexpr uint32_t packed_subordinate_core_coordinates = get_compile_time_arg_val(4);

    // Runtime arguments
    uint32_t receiver_x_coord = packed_subordinate_core_coordinates >> 16;
    uint32_t receiver_y_coord = packed_subordinate_core_coordinates & 0xFFFF;

    uint64_t dst_noc_addr = get_noc_addr(receiver_x_coord, receiver_y_coord, l1_local_read_addr);

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc_async_read(dst_noc_addr, l1_local_read_addr, bytes_per_transaction);
        }
        noc_async_read_barrier();
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Test id", test_id);
}
