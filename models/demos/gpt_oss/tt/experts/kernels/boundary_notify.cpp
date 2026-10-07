// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Decode layer boundary fused into its consumer op (tt/decode_boundary.py: DecodeBoundary.consumer_parts), BRISC of
// the boundary core: once the boundary compute has packed the normed output x (cb_x, backed by the x tensor), increment
// program semaphore `sem_id` on every consumer core; their writers wait for it before reading x.
//
// runtime args: [noc_x, noc_y] per consumer core

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t tiles = get_compile_time_arg_val(1);
    constexpr uint32_t consumers = get_compile_time_arg_val(2);
    constexpr uint32_t sem_id = get_compile_time_arg_val(3);

    cb_wait_front(cb_x, tiles);
    const uint32_t sem = get_semaphore(sem_id);
    for (uint32_t i = 0; i < consumers; ++i) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(2 * i), get_arg_val<uint32_t>(2 * i + 1), sem), 1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_x, tiles);
}
