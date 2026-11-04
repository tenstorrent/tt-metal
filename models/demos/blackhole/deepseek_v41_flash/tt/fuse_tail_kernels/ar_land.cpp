// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// All-reduce landing core (NCRISC): waits until its chunk of the reduced result has arrived (global semaphore), then
// re-arms it.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t gsem = get_common_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gsem);
    {
        DeviceZoneScopedN("L_WAIT");
        noc_semaphore_wait_min(sem, 1);
    }
    noc_semaphore_set(sem, 0);
}
