// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    const uint32_t test_id = get_compile_time_arg_val(0);
    const uint32_t num_writes = get_compile_time_arg_val(1);
    const uint32_t write_value_base = get_compile_time_arg_val(2);
    const uint32_t use_posted_writes = get_compile_time_arg_val(3);
    const uint32_t same_destination = get_compile_time_arg_val(4);
    const uint32_t same_value = get_compile_time_arg_val(5);
    const uint32_t dest_l1_addr = get_compile_time_arg_val(6);
    const uint32_t addr_stride = get_compile_time_arg_val(7);
    const uint32_t packed_receiver_coords = get_compile_time_arg_val(8);
    const uint32_t noc_id = get_compile_time_arg_val(9);

    // Extract receiver coordinates
    uint32_t receiver_x = (packed_receiver_coords >> 16) & 0xFFFF;
    uint32_t receiver_y = packed_receiver_coords & 0xFFFF;

    {
        DeviceZoneScopedN("RISCV0");

        if (same_destination) {
            uint64_t dest_noc_addr = get_noc_addr(receiver_x, receiver_y, dest_l1_addr);

            // Setup state once
            if (same_value) {
                // When writing same value, set it in the state and reuse it
                if (use_posted_writes) {
                    noc_inline_dw_write_set_state<true, true>(
                        dest_noc_addr,
                        write_value_base,  // Set the value once in state
                        0xF,
                        write_at_cmd_buf,
                        noc_id);
                } else {
                    noc_inline_dw_write_set_state<false, true>(
                        dest_noc_addr, write_value_base, 0xF, write_at_cmd_buf, noc_id);
                }

                // Perform writes - reuse value from state
                for (uint32_t i = 0; i < num_writes; i++) {
                    if (use_posted_writes) {
                        noc_inline_dw_write_with_state<false, true, true, false, false>(
                            0,  // value unused since we reuse from state
                            0,
                            write_at_cmd_buf,
                            noc_id);
                    } else {
                        noc_inline_dw_write_with_state<false, true, false, false, false>(
                            0, 0, write_at_cmd_buf, noc_id);
                    }
                }
            } else {
                // When writing different values, set addr only in state
                if (use_posted_writes) {
                    noc_inline_dw_write_set_state<true, false>(
                        dest_noc_addr,
                        0,  // value will be provided in each with_state call
                        0xF,
                        write_at_cmd_buf,
                        noc_id);
                } else {
                    noc_inline_dw_write_set_state<false, false>(dest_noc_addr, 0, 0xF, write_at_cmd_buf, noc_id);
                }

                // Perform writes - provide new value each time
                for (uint32_t i = 0; i < num_writes; i++) {
                    if (use_posted_writes) {
                        noc_inline_dw_write_with_state<false, true, true, false, true>(
                            write_value_base + i, 0, write_at_cmd_buf, noc_id);
                    } else {
                        noc_inline_dw_write_with_state<false, true, false, false, true>(
                            write_value_base + i, 0, write_at_cmd_buf, noc_id);
                    }
                }
            }

        } else {
            // Case 2: Different destinations - update local address each time
            uint64_t base_noc_addr = get_noc_addr(receiver_x, receiver_y, dest_l1_addr);

            if (same_value) {
                // When writing same value to different addresses, set value in state
                if (use_posted_writes) {
                    noc_inline_dw_write_set_state<true, true>(
                        base_noc_addr, write_value_base, 0xF, write_at_cmd_buf, noc_id);
                } else {
                    noc_inline_dw_write_set_state<false, true>(
                        base_noc_addr, write_value_base, 0xF, write_at_cmd_buf, noc_id);
                }

                // Perform writes - provide new address each time, reuse value from state
                for (uint32_t i = 0; i < num_writes; i++) {
                    uint32_t current_local_addr = dest_l1_addr + (i * addr_stride);

                    if (use_posted_writes) {
                        noc_inline_dw_write_with_state<true, true, true, false, false>(
                            0, current_local_addr, write_at_cmd_buf, noc_id);
                    } else {
                        noc_inline_dw_write_with_state<true, true, false, false, false>(
                            0, current_local_addr, write_at_cmd_buf, noc_id);
                    }
                }
            } else {
                // When writing different values to different addresses, set base addr in state
                if (use_posted_writes) {
                    noc_inline_dw_write_set_state<true, false>(base_noc_addr, 0, 0xF, write_at_cmd_buf, noc_id);
                } else {
                    noc_inline_dw_write_set_state<false, false>(base_noc_addr, 0, 0xF, write_at_cmd_buf, noc_id);
                }

                // Perform writes - provide new address and new value each time
                for (uint32_t i = 0; i < num_writes; i++) {
                    uint32_t current_local_addr = dest_l1_addr + (i * addr_stride);

                    if (use_posted_writes) {
                        noc_inline_dw_write_with_state<true, true, true, false, true>(
                            write_value_base + i, current_local_addr, write_at_cmd_buf, noc_id);
                    } else {
                        noc_inline_dw_write_with_state<true, true, false, false, true>(
                            write_value_base + i, current_local_addr, write_at_cmd_buf, noc_id);
                    }
                }
            }
        }

        // Wait for all writes to complete
        noc_async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Stateful", 1);
    DeviceTimestampedData("Posted writes", use_posted_writes);
    DeviceTimestampedData("Number of transactions", num_writes);
    DeviceTimestampedData("Transaction size in bytes", 32);
    DeviceTimestampedData("Same destination", same_destination);
    DeviceTimestampedData("Same value", same_value);
    DeviceTimestampedData("NOC Index", noc_id);
}
