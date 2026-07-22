// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    constexpr uint32_t l1_local_addr = get_arg(args::l1_addr);
    constexpr uint32_t num_transactions = get_arg(args::num_tx);
    constexpr uint32_t transaction_size = get_arg(args::tx_size);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t packed_dest_core_start = get_arg(args::dest_coords);
    constexpr uint32_t packed_dest_core_end = get_arg(args::dest_coords_end);
    constexpr uint32_t loopback = get_arg(args::loopback);
    constexpr uint32_t num_dests = get_arg(args::num_cores);

    uint32_t dest_x_start = packed_dest_core_start >> 16;
    uint32_t dest_y_start = packed_dest_core_start & 0xFFFF;
    uint32_t dest_x_end = packed_dest_core_end >> 16;
    uint32_t dest_y_end = packed_dest_core_end & 0xFFFF;

    if (noc_index == 1) {
        std::swap(dest_x_start, dest_x_end);
        std::swap(dest_y_start, dest_y_end);
    }

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;
    MulticastEndpoint mcast_endpoint;
    constexpr NocOptions mcast_opts = loopback ? NocOptions::MCAST_INCL_SRC : NocOptions::DEFAULT;
    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_transactions; i++) {
            noc.async_write_multicast<mcast_opts>(
                unicast_endpoint,
                mcast_endpoint,
                transaction_size,
                num_dests,
                {.addr = l1_local_addr},
                {.noc_x_start = dest_x_start,
                 .noc_y_start = dest_y_start,
                 .noc_x_end = dest_x_end,
                 .noc_y_end = dest_y_end,
                 .addr = l1_local_addr});
        }
    }

    noc.async_write_barrier();

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc.get_noc_id());
    DeviceTimestampedData("Number of transactions", num_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size);
}
