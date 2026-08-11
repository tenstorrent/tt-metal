// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"

// L1 to L1 all-from-all read (Metal 2.0)
void kernel_main() {
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t mst_l1_base_address = get_arg(args::mst_l1_addr);
    constexpr uint32_t sub_l1_base_address = get_arg(args::sub_l1_addr);
    constexpr uint32_t num_subordinates = get_arg(args::num_subordinates);
    constexpr uint32_t num_virtual_channels = get_arg(args::num_vc);

    const uint32_t num_of_transactions = get_arg(args::num_of_transactions);
    const uint32_t bytes_per_transaction_per_subordinate = get_arg(args::bytes_per_transaction);

    std::array<std::array<uint32_t, 2>, num_subordinates> subordinate_coords;
    uint32_t rt_args_idx = 0;
    for (uint32_t i = 0; i < num_subordinates; i++) {
        subordinate_coords[i][0] = get_vararg(rt_args_idx++);
        subordinate_coords[i][1] = get_vararg(rt_args_idx++);
    }

    uint32_t master_l1_local_address = mst_l1_base_address;
    uint32_t subordinate_l1_local_address = sub_l1_base_address;

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t j = 0; j < num_subordinates; j++) {
            uint32_t subordinate_x_coord = subordinate_coords[j][0];
            uint32_t subordinate_y_coord = subordinate_coords[j][1];

            for (uint32_t i = 0; i < num_of_transactions; i++) {
                uint32_t current_virtual_channel = i % num_virtual_channels;

                noc.async_read<NocOptions::CUSTOM_VC>(
                    unicast_endpoint,
                    unicast_endpoint,
                    bytes_per_transaction_per_subordinate,
                    {
                        .noc_x = subordinate_x_coord,
                        .noc_y = subordinate_y_coord,
                        .addr = subordinate_l1_local_address,
                    },
                    {
                        .addr = master_l1_local_address,
                    },
                    NocOptVals{.vc = current_virtual_channel});
            }
        }
        noc.async_read_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions * num_subordinates);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction_per_subordinate);
    DeviceTimestampedData("NoC Index", noc.get_noc_id());
    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
    DeviceTimestampedData("Master Grid Size X", get_arg(args::mst_grid_size_x));
    DeviceTimestampedData("Master Grid Size Y", get_arg(args::mst_grid_size_y));
    DeviceTimestampedData("Subordinate Grid Size X", get_arg(args::sub_grid_size_x));
    DeviceTimestampedData("Subordinate Grid Size Y", get_arg(args::sub_grid_size_y));
}
