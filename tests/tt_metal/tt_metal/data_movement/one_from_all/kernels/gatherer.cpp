// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 request
void kernel_main() {
    uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t total_subordinate_cores = get_compile_time_arg_val(4);

    std::array<uint32_t, total_subordinate_cores> subordinate_l1_byte_addresses;
    std::array<std::array<uint32_t, 2>, total_subordinate_cores> responder_coords;
    uint32_t rt_args_idx = 0;
    for (uint32_t i = 0; i < total_subordinate_cores; i++) {
        subordinate_l1_byte_addresses[i] = l1_local_addr;
        responder_coords[i][0] = get_arg_val<uint32_t>(rt_args_idx++);
        responder_coords[i][1] = get_arg_val<uint32_t>(rt_args_idx++);
    }

    constexpr uint32_t subordinate_size_bytes = num_of_transactions * transaction_size_bytes;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes * total_subordinate_cores);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t sub_core = 0; sub_core < total_subordinate_cores; sub_core++) {
            uint64_t src_base_noc_addr = get_noc_addr(responder_coords[sub_core][0], responder_coords[sub_core][1], 0);
            for (uint32_t i = 0; i < num_of_transactions; i++) {
                /* The 64-bit NOC addresses consists of a 32-bit local address and a NOC XY coordinate. The local
                 * address occupies the lower 32 bits and the NOC XY coordinate occupies the next 12 (unicast) to 24
                 * (multicast) bits. In the get_noc_addr call, we set the local address to 0 to get the base address.
                 * Then, we OR it with the local address in each iteration to get the full NOC address. */
                uint64_t src_noc_addr = src_base_noc_addr | subordinate_l1_byte_addresses[sub_core];

                noc_async_read(src_noc_addr, l1_local_addr, transaction_size_bytes);

                subordinate_l1_byte_addresses[sub_core] += transaction_size_bytes;
                l1_local_addr += transaction_size_bytes;
            }
        }
        noc_async_read_barrier();
    }
}
