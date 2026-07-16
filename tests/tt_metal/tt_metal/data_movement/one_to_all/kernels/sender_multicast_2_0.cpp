// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "api/dataflow/endpoints.h"
#include "api/debug/dprint.h"

// L1 to L1 send
void kernel_main() {
    constexpr uint32_t mst_base_addr = get_arg(args::mst_base_addr);
    constexpr uint32_t sub_base_addr = get_arg(args::sub_base_addr);
    constexpr uint32_t bytes_per_page = get_arg(args::bytes_per_page);
    constexpr uint32_t test_id = get_arg(args::test_id);
    constexpr uint32_t num_subordinates = get_arg(args::num_subordinates);
    constexpr bool is_linked = get_arg(args::is_linked);
    constexpr bool loopback = get_arg(args::loopback);
    uint32_t start_x = get_arg(args::start_x);
    uint32_t start_y = get_arg(args::start_y);
    uint32_t end_x = get_arg(args::end_x);
    uint32_t end_y = get_arg(args::end_y);

    // Specific for Multicast Schemes
    constexpr uint32_t multicast_scheme_type = get_arg(args::mcast_scheme_type);
    constexpr uint32_t sub_grid_size_x = get_arg(args::sub_grid_size_x);
    constexpr uint32_t sub_grid_size_y = get_arg(args::sub_grid_size_y);

    const uint32_t num_of_transactions = get_arg(args::num_of_transactions);
    const uint32_t pages_per_transaction = get_arg(args::pages_per_transaction);
    const uint32_t bytes_per_transaction = pages_per_transaction * bytes_per_page;

    if (noc_index == 1) {
        std::swap(start_x, end_x);
        std::swap(start_y, end_y);
    }

    Noc noc(noc_index);
    UnicastEndpoint unicast_endpoint;
    MulticastEndpoint multicast_endpoint;

    constexpr NocOptions mcast_opts = loopback ? NocOptions::MCAST_INCL_SRC : NocOptions::DEFAULT;

    auto do_mcast = [&](bool linked) {
        noc.async_write_multicast<mcast_opts>(
            unicast_endpoint,
            multicast_endpoint,
            bytes_per_transaction,
            num_subordinates,
            {.addr = mst_base_addr},
            {
                .noc_x_start = start_x,
                .noc_y_start = start_y,
                .noc_x_end = end_x,
                .noc_y_end = end_y,
                .addr = sub_base_addr,
            },
            linked);
    };

    {
        DeviceZoneScopedN("RISCV0");

        for (uint32_t i = 0; i < num_of_transactions - 1; i++) {
            do_mcast(is_linked);
        }
        // Last packet is sent separately to unlink the transaction,
        // so the next one can use the VC and do its own path reservation
        do_mcast(false);
        noc.async_write_barrier();
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc_index);

    // For multicast schemes, we can also log the multicast scheme type and grid size
    if constexpr (multicast_scheme_type != 0) {
        DeviceTimestampedData("Multicast Scheme Type", multicast_scheme_type);
        DeviceTimestampedData("Subordinate Grid Size X", sub_grid_size_x);
        DeviceTimestampedData("Subordinate Grid Size Y", sub_grid_size_y);
    }

    DeviceTimestampedData("Number of subordinates", num_subordinates);
    DeviceTimestampedData("Loopback", loopback ? 1 : 0);
}
