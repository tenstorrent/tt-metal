// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    auto* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    auto* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    noc_semaphore_wait(credit, 1);
    noc_semaphore_set(ready, 1);
    const uint64_t remote = get_noc_addr(get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_semaphore(1));
    noc_semaphore_set_remote(reinterpret_cast<uint32_t>(ready), remote);
    noc_async_write_barrier();
}
