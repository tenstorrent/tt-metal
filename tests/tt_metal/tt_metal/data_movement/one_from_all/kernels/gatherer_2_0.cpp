// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

// L1 to L1 request (Metal 2.0)
void kernel_main() {
    constexpr uint32_t l1_local_addr = get_arg(args::l1_addr);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t total_subordinate_cores = get_arg(args::num_subordinates);
    constexpr uint32_t num_virtual_channels = get_arg(args::num_vc);

    // varargs: [0]=num_transactions, [1]=transaction_size_bytes,
    const uint32_t num_of_transactions = get_arg(args::num_of_transactions);
    const uint32_t transaction_size_bytes = get_arg(args::transaction_size_bytes);

    std::array<std::array<uint32_t, 2>, total_subordinate_cores> responder_coords;
    uint32_t rt_args_idx = 0;
    for (uint32_t i = 0; i < total_subordinate_cores; i++) {
        responder_coords[i][0] = get_vararg(rt_args_idx++);
        responder_coords[i][1] = get_vararg(rt_args_idx++);
    }

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t sub_core = 0; sub_core < total_subordinate_cores; sub_core++) {
            for (uint32_t i = 0; i < num_of_transactions; i++) {
                uint32_t current_virtual_channel = i % num_virtual_channels;
                noc.async_read<NocOptions::CUSTOM_VC>(
                    unicast_endpoint,
                    unicast_endpoint,
                    transaction_size_bytes,
                    {.noc_x = responder_coords[sub_core][0],
                     .noc_y = responder_coords[sub_core][1],
                     .addr = l1_local_addr},
                    {.addr = l1_local_addr},
                    NocOptVals{.vc = current_virtual_channel});
            }
        }
        noc.async_read_barrier();
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("Number of transactions", num_of_transactions * total_subordinate_cores);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("NoC Index", noc_index);
    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
    DeviceTimestampedData("Number of subordinates", total_subordinate_cores);
}
