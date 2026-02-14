// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"

void kernel_main() {
    constexpr uint32_t l1_local_addr = get_compile_time_arg_val(0);
    constexpr uint32_t num_transactions = get_compile_time_arg_val(1);
    constexpr uint32_t transaction_size = get_compile_time_arg_val(2);
    constexpr uint32_t test_id = get_compile_time_arg_val(3);
    constexpr uint32_t packed_dest_core_start = get_compile_time_arg_val(4);
    constexpr uint32_t packed_dest_core_end = get_compile_time_arg_val(5);
    constexpr uint32_t loopback = get_compile_time_arg_val(6);
    constexpr uint32_t num_dests = get_compile_time_arg_val(7);

    uint32_t dest_x_start = packed_dest_core_start >> 16;
    uint32_t dest_y_start = packed_dest_core_start & 0xFFFF;
    uint32_t dest_x_end = packed_dest_core_end >> 16;
    uint32_t dest_y_end = packed_dest_core_end & 0xFFFF;

    if (noc_index == 1) {
        std::swap(dest_x_start, dest_x_end);
        std::swap(dest_y_start, dest_y_end);
    }

    experimental::Noc noc(noc_index);
    experimental::UnicastEndpoint unicast_endpoint;
    experimental::MulticastEndpoint mcast_endpoint;
    constexpr auto mcast_mode =
        loopback ? experimental::Noc::McastMode::INCLUDE_SRC : experimental::Noc::McastMode::EXCLUDE_SRC;
    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_transactions; i++) {
            noc.async_write_multicast<mcast_mode>(
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
