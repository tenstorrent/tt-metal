// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "ckernel.h"

void kernel_main() {
    // Compile-time arguments
    const uint32_t test_id = get_compile_time_arg_val(0);
    const uint32_t l1_base_address = get_compile_time_arg_val(1);
    const uint32_t total_size_bytes_per_master = get_compile_time_arg_val(2);
    const uint32_t num_of_transactions = get_compile_time_arg_val(3);
    const uint32_t bytes_per_transaction_per_master = get_compile_time_arg_val(4);
    const uint32_t num_subordinates = get_compile_time_arg_val(5);

    // Runtime arguments
    const uint32_t master_index = get_arg_val<uint32_t>(0);

    // Derivative values
    uint32_t master_l1_local_address = l1_base_address;
    uint32_t subordinate_l1_local_address = l1_base_address + (master_index * total_size_bytes_per_master);

    uint32_t subordinate_x_coord;
    uint32_t subordinate_y_coord;

    uint64_t subordinate_l1_noc_address;

    uint32_t start_time = ckernel::read_wall_clock();

    {
        DeviceZoneScopedN("RISCV0");

        for (uint32_t i = 0; i < num_of_transactions; i++) {
            for (uint32_t j = 0; j < num_subordinates; j++) {
                subordinate_x_coord = get_arg_val<uint32_t>(1 + (j * 2));
                subordinate_y_coord = get_arg_val<uint32_t>(1 + (j * 2) + 1);

                subordinate_l1_noc_address =
                    get_noc_addr(subordinate_x_coord, subordinate_y_coord, subordinate_l1_local_address);

                noc_async_write(master_l1_local_address, subordinate_l1_noc_address, bytes_per_transaction_per_master);
            }
            master_l1_local_address += bytes_per_transaction_per_master;
            subordinate_l1_local_address += bytes_per_transaction_per_master;
        }
        noc_async_write_barrier();
    }

    uint32_t end_time = ckernel::read_wall_clock();

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction_per_master * num_subordinates);
    DeviceTimestampedData(
        "Total bytes transferred", num_of_transactions * bytes_per_transaction_per_master * num_subordinates);
    DeviceTimestampedData("Test id", test_id);

    DPRINT << "Sender kernel completed in " << (end_time - start_time) << " cycles." << ENDL();
}
