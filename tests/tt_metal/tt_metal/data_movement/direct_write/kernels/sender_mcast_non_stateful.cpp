// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "hw/inc/api/debug/dprint.h"

void kernel_main() {
#ifdef ARCH_QUASAR
    // Quasar: use named compile-time args (Metal 2.0 API)
    constexpr uint32_t test_id = get_named_compile_time_arg_val("test_id");
    constexpr uint32_t num_writes = get_named_compile_time_arg_val("num_writes");
    constexpr uint32_t sub_base_addr = get_named_compile_time_arg_val("sub_base_addr");
    constexpr uint32_t write_value_base = get_named_compile_time_arg_val("write_val_base");
    constexpr uint32_t same_destination = get_named_compile_time_arg_val("same_dest");
    constexpr uint32_t addr_stride = get_named_compile_time_arg_val("addr_stride");
    constexpr uint32_t noc_index = get_named_compile_time_arg_val("noc_index");
    constexpr uint32_t num_subordinates = get_named_compile_time_arg_val("num_subordinates");
    constexpr uint32_t start_x = get_named_compile_time_arg_val("start_x");
    constexpr uint32_t start_y = get_named_compile_time_arg_val("start_y");
    constexpr uint32_t end_x = get_named_compile_time_arg_val("end_x");
    constexpr uint32_t end_y = get_named_compile_time_arg_val("end_y");
#else
    // WH/BH: use indexed compile-time args (legacy API)
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t num_writes = get_compile_time_arg_val(1);
    constexpr uint32_t sub_base_addr = get_compile_time_arg_val(2);
    constexpr uint32_t write_value_base = get_compile_time_arg_val(3);
    constexpr uint32_t same_destination = get_compile_time_arg_val(4);
    constexpr uint32_t addr_stride = get_compile_time_arg_val(5);
    constexpr uint32_t noc_index = get_compile_time_arg_val(6);
    constexpr uint32_t num_subordinates = get_compile_time_arg_val(7);
    constexpr uint32_t start_x = get_compile_time_arg_val(8);
    constexpr uint32_t start_y = get_compile_time_arg_val(9);
    constexpr uint32_t end_x = get_compile_time_arg_val(10);
    constexpr uint32_t end_y = get_compile_time_arg_val(11);
#endif

    {
        DeviceZoneScopedN("RISCV0");

        // Create multicast address for the rectangle of destinations
        uint64_t dst_noc_addr_multicast = noc_index == 0
                                              ? get_noc_multicast_addr(start_x, start_y, end_x, end_y, sub_base_addr)
                                              : get_noc_multicast_addr(end_x, end_y, start_x, start_y, sub_base_addr);

        for (uint32_t i = 0; i < num_writes; i++) {
            uint32_t write_value = write_value_base + i;
            noc_inline_mcast_dw_write<InlineWriteDst::DEFAULT, true, true>(
                dst_noc_addr_multicast, write_value, 0xF, noc_index, NOC_MULTICAST_WRITE_VC, 0, num_subordinates);
            if constexpr (!same_destination) {
                dst_noc_addr_multicast += addr_stride;
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
    DeviceTimestampedData("Multicast", 1);
    DeviceTimestampedData("NoC Index", noc_index);
}
