// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Test helper: spins for runtime-arg microseconds (skews the chips of a mesh against each other).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t us = get_arg_val<uint32_t>(0);
    const uint64_t t0 = get_timestamp();
    while (get_timestamp() - t0 < (uint64_t)us * 1350) {
    }
}
