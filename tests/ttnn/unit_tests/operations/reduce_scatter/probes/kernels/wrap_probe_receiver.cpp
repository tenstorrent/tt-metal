// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Refinement-1 fabric probe (receiver, NCRISC): wait for the sender's counting inc (which lands
// AFTER the page data, in-order on the fabric connection), then re-arm the semaphore. The program
// completing is the host's signal that the page landed.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t counting_sem_addr = get_arg_val<uint32_t>(0);
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counting_sem_addr);
    noc_semaphore_wait_min(sem_ptr, 1);
    noc_semaphore_set(sem_ptr, 0);  // cache-reuse re-arm
}
