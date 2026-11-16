// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 router projection, reducer reader (NCRISC) (fused_decode/router.py: RouterB1): waits until the `cores`
// matmul cores wrote row 0 of their partial tiles (one semaphore increment each), then publishes the partial tiles
// to the compute. Rows 1..31 of the partial tiles are never written and never read back.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);
    constexpr uint32_t cores = get_compile_time_arg_val(1);
    constexpr uint32_t sem_id = get_compile_time_arg_val(2);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    noc_semaphore_wait(sem, cores);
    noc_semaphore_set(sem, 0);
    cb_reserve_back(cb_part, cores);
    cb_push_back(cb_part, cores);
}
