// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x52 streamed decode matmul, idle core of the multicast rectangle (the rectangle is larger than the number of columns
// groups): the hub's semaphore multicast reaches it too, so it must consume and clear it like the working cores do, or
// the next program that gives this core real work would find the semaphore already set and start on a stale activation.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc_semaphore.h"

void kernel_main() {
    constexpr uint32_t sem_id = get_named_compile_time_arg_val("sem_x");
    noc_semaphore_inc(
        get_noc_addr(
            get_arg_val<uint32_t>(0),
            get_arg_val<uint32_t>(1),
            get_semaphore(get_named_compile_time_arg_val("sem_hub"))),
        1);
    Semaphore<> sem(sem_id);
    sem.wait_min(1);
    sem.set(0);
}
