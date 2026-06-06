// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t expected_value = get_arg(args::expected_value);
    constexpr uint32_t test_id = get_arg(args::test_id);

    Noc noc(noc_index);
    Semaphore semaphore(sem::sem_name);

    {
        DeviceZoneScopedN("RISCV1");
        semaphore.wait(expected_value);
    }

    DeviceTimestampedData("Number of transactions", 1);
    DeviceTimestampedData("Transaction size in bytes", 1);
    DeviceTimestampedData("Test id", test_id);
}
