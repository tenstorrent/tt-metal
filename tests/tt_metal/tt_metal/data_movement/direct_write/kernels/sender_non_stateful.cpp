// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
#ifdef ARCH_QUASAR
    // Quasar: use named compile-time args (Metal 2.0 API)
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t num_writes = get_named_compile_time_arg_val("num_writes");
    constexpr uint32_t write_value_base = get_named_compile_time_arg_val("write_val_base");
    constexpr uint32_t use_posted_writes = get_named_compile_time_arg_val("use_posted");
    constexpr uint32_t same_destination = get_named_compile_time_arg_val("same_dest");
    constexpr uint32_t same_value = get_named_compile_time_arg_val("same_value");
    constexpr uint32_t dest_l1_addr = get_named_compile_time_arg_val("dest_l1_addr");
    constexpr uint32_t addr_stride = get_named_compile_time_arg_val("addr_stride");
    constexpr uint32_t packed_receiver_coords = get_named_compile_time_arg_val("receiver_coords");
    constexpr uint32_t noc_id = get_named_compile_time_arg_val("noc_id");
#else
    // WH/BH: use indexed compile-time args (legacy API)
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_writes = get_compile_time_arg_val(1);
    constexpr uint32_t write_value_base = get_compile_time_arg_val(2);
    constexpr uint32_t use_posted_writes = get_compile_time_arg_val(3);
    constexpr uint32_t same_destination = get_compile_time_arg_val(4);
    constexpr uint32_t same_value = get_compile_time_arg_val(5);
    constexpr uint32_t dest_l1_addr = get_compile_time_arg_val(6);
    constexpr uint32_t addr_stride = get_compile_time_arg_val(7);
    constexpr uint32_t packed_receiver_coords = get_compile_time_arg_val(8);
    constexpr uint32_t noc_id = get_compile_time_arg_val(9);
#endif

    // Extract receiver coordinates
    uint32_t receiver_x = (packed_receiver_coords >> 16) & 0xFFFF;
    uint32_t receiver_y = packed_receiver_coords & 0xFFFF;

    {
        DeviceZoneScopedN("RISCV0");
        if constexpr (same_destination) {
            // Case 1: All writes to same destination - non-stateful reconfigures each time
            uint64_t dest_noc_addr = get_noc_addr(receiver_x, receiver_y, dest_l1_addr);

            for (uint32_t i = 0; i < num_writes; i++) {
                uint32_t write_value = same_value ? write_value_base : (write_value_base + i);
                if constexpr (use_posted_writes) {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(dest_noc_addr, write_value, 0xF, noc_id);
                } else {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, false>(dest_noc_addr, write_value, 0xF, noc_id);
                }
            }

        } else {
            // Case 2: Different destinations - each write has different address
            for (uint32_t i = 0; i < num_writes; i++) {
                uint32_t current_local_addr = dest_l1_addr + (i * addr_stride);
                uint64_t dest_noc_addr = get_noc_addr(receiver_x, receiver_y, current_local_addr);
                uint32_t write_value = same_value ? write_value_base : (write_value_base + i);
                if constexpr (use_posted_writes) {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, true>(dest_noc_addr, write_value, 0xF, noc_id);
                } else {
                    noc_inline_dw_write<InlineWriteDst::DEFAULT, false>(dest_noc_addr, write_value, 0xF, noc_id);
                }
            }
        }

        // Wait for all writes to complete
        noc_async_write_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Stateful", 0);
    DeviceTimestampedData("Posted writes", use_posted_writes);
    DeviceTimestampedData("Number of transactions", num_writes);
    DeviceTimestampedData("Transaction size in bytes", 32);
    DeviceTimestampedData("Same destination", same_destination);
    DeviceTimestampedData("Same value", same_value);
    DeviceTimestampedData("NoC Index", noc_id);
}
