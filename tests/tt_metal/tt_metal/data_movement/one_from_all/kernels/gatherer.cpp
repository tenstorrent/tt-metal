// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 request
void kernel_main() {
    uint32_t dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_num_pages = get_compile_time_arg_val(2);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t test_id = get_compile_time_arg_val(4);
    constexpr uint32_t total_subordinate_cores = get_compile_time_arg_val(5);

    std::array<uint32_t, total_subordinate_cores> subordinate_l1_byte_addresses;
    std::array<std::array<uint32_t, 2>, total_subordinate_cores> responder_coords;
    uint32_t rt_args_idx = 0;
    for (uint32_t i = 0; i < total_subordinate_cores; i++) {
        subordinate_l1_byte_addresses[i] = get_arg_val<uint32_t>(rt_args_idx++);
        responder_coords[i][0] = get_arg_val<uint32_t>(rt_args_idx++);
        responder_coords[i][1] = get_arg_val<uint32_t>(rt_args_idx++);
    }

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;
    constexpr uint32_t subordinate_size_bytes = num_of_transactions * transaction_num_pages * page_size_bytes;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes * total_subordinate_cores);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            uint32_t subordinate_addr_offset = 0;
            for (uint32_t j = 0; j < total_subordinate_cores; j++) {
                uint64_t src_noc_addr =
                    get_noc_addr(responder_coords[j][0], responder_coords[j][1], subordinate_l1_byte_addresses[j]);

                noc_async_read(src_noc_addr, dst_addr + subordinate_addr_offset, transaction_size_bytes);

                subordinate_l1_byte_addresses[j] += transaction_size_bytes;
                subordinate_addr_offset += subordinate_size_bytes;
            }
            dst_addr += transaction_size_bytes;
        }
        noc_async_read_barrier();
    }
}
