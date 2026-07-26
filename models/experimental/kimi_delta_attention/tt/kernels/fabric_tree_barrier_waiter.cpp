// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Local endpoint of one fabric-tree barrier edge.  The semaphore is reset on
// every invocation so the cached generic program is safe for repeated eager
// dispatches and trace replay.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t semaphore_addr = get_arg_val<uint32_t>(0);
    auto* semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    noc_semaphore_wait_min(semaphore, 1);
    noc_semaphore_set(semaphore, 0);
}
