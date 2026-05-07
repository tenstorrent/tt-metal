// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"
#include "experimental/noc_semaphore.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"

// L1 to L1 send
void kernel_main() {
    constexpr uint32_t src_addr = get_arg(args::src_addr);
    constexpr uint32_t dst_addr = get_arg(args::dst_addr);
    constexpr uint32_t num_of_transactions = get_arg(args::num_transactions);
    constexpr uint32_t transaction_num_pages = get_arg(args::tx_num_pages);
    constexpr uint32_t page_size_bytes = get_arg(args::page_size);
    constexpr uint32_t test_id = get_arg(args::test_id);

    experimental::Semaphore semaphore(sem::sync_sem);
    uint32_t dest_x = get_vararg(0);
    uint32_t dest_y = get_vararg(1);

    constexpr uint32_t transaction_size_bytes = transaction_num_pages * page_size_bytes;

    experimental::Noc noc;
    experimental::UnicastEndpoint unicast_endpoint;

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", transaction_size_bytes);
    DeviceTimestampedData("Test id", test_id);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            noc.async_write(
                unicast_endpoint,
                unicast_endpoint,
                transaction_size_bytes,
                {.addr = src_addr},
                {.noc_x = dest_x, .noc_y = dest_y, .addr = dst_addr});
        }
        noc.async_write_barrier();
    }

    semaphore.up(noc, dest_x, dest_y, 1);
}
