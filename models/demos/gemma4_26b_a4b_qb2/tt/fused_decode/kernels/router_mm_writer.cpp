// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 router projection, matmul core writer (BRISC) (fused_decode/router.py: RouterB1): row 0 of the FP32 partial
// tile (face 0 row 0 and face 1 row 0, 64 B each) into row 0 of the reducer's partial tile `slot` (same offsets), then
// one increment of the reducer's semaphore.
//
// runtime args: [reducer_noc_x, reducer_noc_y, slot]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t rx = get_arg_val<uint32_t>(0);
    const uint32_t ry = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_part = get_compile_time_arg_val(1);  // the reducer's partial tiles: same address everywhere
    constexpr uint32_t sem_id = get_compile_time_arg_val(2);

    cb_wait_front(cb_out, 1);
    const uint32_t src = get_read_ptr(cb_out);
    const uint32_t dst = get_write_ptr(cb_part) + 4096 * slot;
    noc_async_write(src, get_noc_addr(rx, ry, dst), 64);
    noc_async_write(src + 1024, get_noc_addr(rx, ry, dst + 1024), 64);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(rx, ry, get_semaphore(sem_id)), 1);
    noc_async_atomic_barrier();
    cb_pop_front(cb_out, 1);
}
