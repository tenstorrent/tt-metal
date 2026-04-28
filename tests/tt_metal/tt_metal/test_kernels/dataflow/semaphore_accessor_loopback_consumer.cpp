// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Minimal test kernel: waits on a semaphore via its sem:: accessor name.
// The consumer's accessor name differs from the producer's to prove both kernels
// independently resolve their names to the same underlying sem ID.

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc_semaphore.h"

void kernel_main() {
    experimental::Semaphore s(sem::waiter);
    s.wait(1);
}
