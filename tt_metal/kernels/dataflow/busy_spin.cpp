// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Busy-spin kernel for async-dispatch teardown race condition tests.
// Accepts a single runtime arg: number of iterations to spin.
// Used to guarantee the kernel is still executing when the host calls close(),
// so that async-teardown tests create a real race rather than a clean teardown.
//
// At WH BRISC ~1.2 GHz with volatile load+store (~10 cycles/iter):
//   1,000,000 iterations ≈ 8 ms  (safe margin above host teardown overhead ~10 µs)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t iters = get_arg_val<uint32_t>(0);
    for (volatile uint32_t i = 0; i < iters; i++) {}
}
