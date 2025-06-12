// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ckernel.h"

void kernel_main() {
    // Compile-time arguments
    const uint32_t test_id = get_compile_time_arg_val(0);
    const uint32_t mst_l1_base_address = get_compile_time_arg_val(1);
    const uint32_t sub_l1_base_address = get_compile_time_arg_val(2);
    const uint32_t total_size_bytes_per_subordinate = get_compile_time_arg_val(3);
    const uint32_t num_of_transactions = get_compile_time_arg_val(4);
    const uint32_t bytes_per_transaction_per_subordinate = get_compile_time_arg_val(5);
    const uint32_t num_subordinates = get_compile_time_arg_val(6);

    // Runtime arguments
    const uint32_t master_index = get_arg_val<uint32_t>(0);

    // Derivative values
    uint32_t master_l1_local_address = mst_l1_base_address;
    uint32_t subordinate_l1_local_address = sub_l1_base_address;

    uint32_t subordinate_x_coord;
    uint32_t subordinate_y_coord;

    uint64_t subordinate_l1_noc_base_address;
    uint64_t subordinate_l1_noc_address;

    {
        DeviceZoneScopedN("RISCV0");

        // Loop Order 1: Number of transactions -> Number of subordinates

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            master_l1_local_address = mst_l1_base_address + (i * bytes_per_transaction_per_subordinate);

            for (uint32_t j = 0; j < num_subordinates; j++) {
                // Subordinate coordiantes are stored in the runtime arguments starting at index 1
                // Each x coordinate is stores in an odd index
                // Each y coordinate is stored in the next even index
                // The first subordinate's coordinates are at indices 1 and 2, the second at indices 3 and 4, etc.
                subordinate_x_coord = get_arg_val<uint32_t>(1 + (j * 2));
                subordinate_y_coord = get_arg_val<uint32_t>(1 + (j * 2) + 1);

                subordinate_l1_noc_address =
                    get_noc_addr(subordinate_x_coord, subordinate_y_coord, subordinate_l1_local_address);

                noc_async_read(
                    subordinate_l1_noc_address, master_l1_local_address, bytes_per_transaction_per_subordinate);

                master_l1_local_address += total_size_bytes_per_subordinate;
            }

            subordinate_l1_local_address += bytes_per_transaction_per_subordinate;
        }

        /*

        // Loop Order 2: Number of subordinates -> Number of transactions

        for (uint32_t i = 0; i < num_subordinates; i++) {

            subordinate_x_coord = get_arg_val<uint32_t>(1 + (i * 2));
            subordinate_y_coord = get_arg_val<uint32_t>(1 + (i * 2) + 1);

            subordinate_l1_noc_base_address =
                get_noc_addr(subordinate_x_coord, subordinate_y_coord, 0);

            subordinate_l1_local_address = sub_l1_base_address;

            for (uint32_t j = 0; j < num_of_transactions; j++) {

                subordinate_l1_noc_address =
                    subordinate_l1_noc_base_address | subordinate_l1_local_address;

                noc_async_read(subordinate_l1_noc_address, master_l1_local_address,
                                bytes_per_transaction_per_subordinate);

                master_l1_local_address += bytes_per_transaction_per_subordinate;
                subordinate_l1_local_address += bytes_per_transaction_per_subordinate;

                //noc_async_read_barrier(); // 27.7 B/cycle
            }
            //noc_async_read_barrier(); // 27.7 B/cycle
        }

        */

        noc_async_read_barrier();  // 30.63 B/cycle
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction_per_subordinate * num_subordinates);
    DeviceTimestampedData("Total bytes", num_of_transactions * bytes_per_transaction_per_subordinate * num_subordinates)
}
