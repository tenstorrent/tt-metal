// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"

// L1 to L1 request
void kernel_main() {
    uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_size_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);

    uint32_t responder_x_coord = get_arg_val<uint32_t>(0);
    uint32_t responder_y_coord = get_arg_val<uint32_t>(1);

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Total bytes transferred", num_of_transactions * transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV1");
        uint64_t src_base_noc_addr = get_noc_addr(responder_x_coord, responder_y_coord, 0);
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            /* The 64-bit NOC addresses consists of a 32-bit local address and a NOC XY coordinate. The local address
             * occupies the lower 32 bits and the NOC XY coordinate occupies the next 12 (unicast) to 24 (multicast)
             * bits. In the get_noc_addr call, we set the local address to 0 to get the base address. Then, we OR it
             * with the local address (src_addr) in each iteration to get the full NOC address. */
            uint64_t src_noc_addr = src_base_noc_addr | l1_local_addr;

            noc_async_read(src_noc_addr, l1_local_addr, transaction_size_bytes);

            l1_local_addr += transaction_size_bytes;
        }
        noc_async_read_barrier();
    }
}
