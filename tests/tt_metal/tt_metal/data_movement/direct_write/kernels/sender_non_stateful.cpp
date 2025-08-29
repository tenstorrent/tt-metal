// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    // Compile-time arguments - same as stateful version for direct comparison
    const uint32_t test_id = get_compile_time_arg_val(0);
    const uint32_t num_writes = get_compile_time_arg_val(1);
    const uint32_t write_value_base = get_compile_time_arg_val(2);
    const uint32_t use_posted_writes = get_compile_time_arg_val(3);
    const uint32_t same_destination = get_compile_time_arg_val(4);
    const uint32_t dest_l1_addr = get_compile_time_arg_val(5);
    const uint32_t addr_stride = get_compile_time_arg_val(6);
    const uint32_t packed_receiver_coords = get_compile_time_arg_val(7);
    const uint32_t noc_id = get_compile_time_arg_val(8);

    // Extract receiver coordinates
    uint32_t receiver_x = (packed_receiver_coords >> 16) & 0xFFFF;
    uint32_t receiver_y = packed_receiver_coords & 0xFFFF;

    {
        DeviceZoneScopedN("RISCV0");
        if (same_destination) {
            // Case 1: All writes to same destination - non-stateful reconfigures each time
            uint64_t dest_noc_addr = get_noc_addr(receiver_x, receiver_y, dest_l1_addr);

            for (uint32_t i = 0; i < num_writes; i++) {
                if (use_posted_writes) {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(  // posted=true
                        dest_noc_addr,
                        write_value_base + i,
                        0xF,
                        noc_id);
                } else {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, false>(  // posted=false
                        dest_noc_addr,
                        write_value_base + i,
                        0xF,
                        noc_id);
                }
            }

        } else {
            // Case 2: Different destinations - each write has different address
            for (uint32_t i = 0; i < num_writes; i++) {
                uint32_t current_local_addr = dest_l1_addr + (i * addr_stride);
                uint64_t dest_noc_addr = get_noc_addr(receiver_x, receiver_y, current_local_addr);
                if (use_posted_writes) {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(
                        dest_noc_addr, write_value_base + i, 0xF, noc_id);
                } else {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, false>(
                        dest_noc_addr, write_value_base + i, 0xF, noc_id);
                }
            }
        }

        // Wait for all writes to complete
        noc_async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Stateful", 0);  // 0 = non-stateful
    DeviceTimestampedData("Posted writes", use_posted_writes);
    DeviceTimestampedData("Number of transactions", num_writes);
    DeviceTimestampedData("Transaction size in bytes", 32);
    DeviceTimestampedData("Same destination", same_destination);
    DeviceTimestampedData("NOC Index", noc_id);
}
