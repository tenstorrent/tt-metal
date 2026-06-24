// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_of_transactions = get_arg(args::num_of_transactions);
    constexpr uint32_t atomic_inc_value = get_arg(args::atomic_inc_value);
    constexpr uint32_t num_dests = get_arg(args::num_dests);
    constexpr uint32_t test_id = get_arg(args::test_id);

    // varargs: [0]=dst_start_x, [1]=dst_start_y, [2]=dst_end_x, [3]=dst_end_y.
    uint32_t dst_start_x = get_arg(args::dst_start_x);
    uint32_t dst_start_y = get_arg(args::dst_start_y);
    uint32_t dst_end_x = get_arg(args::dst_end_x);
    uint32_t dst_end_y = get_arg(args::dst_end_y);

    // For NOC_1, the coordinate system is inverted, so start/end need to be swapped
    if (noc_index == 1) {
        std::swap(dst_start_x, dst_end_x);
        std::swap(dst_start_y, dst_end_y);
    }

    Noc noc(noc_index);
    Semaphore semaphore(sem::sem_name);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t i = 0; i < num_of_transactions; i++) {
            semaphore.inc_multicast(noc, dst_start_x, dst_start_y, dst_end_x, dst_end_y, atomic_inc_value, num_dests);
        }
        noc.async_atomic_barrier();
    }

    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", 4);
    DeviceTimestampedData("Test id", test_id);
}
