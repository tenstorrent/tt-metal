// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Multi-core decode boundary, helper writer (BRISC): (1) this helper's row sums of squares (one FP32 tile from
// compute) -> slot `index` of cb_xchg on every helper (itself included), then program semaphore XCHG_SEM on each;
// (2) its normed pages stay in its shard of the normed output (cb_x backed by it; the consumers read them from here):
// once they are packed, program semaphore 0 on every consumer core (the consumers wait for one increment per helper).
//
// runtime args: [index, (noc x, y) x helpers, (noc x, y) x consumers (from BND_DIRECT_ARG; consumer_parts)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t index = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xchg = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = get_compile_time_arg_val(2);
    constexpr uint32_t PH = get_compile_time_arg_val(3);
    constexpr uint32_t helpers = get_compile_time_arg_val(4);
    constexpr uint32_t xchg_sem = get_compile_time_arg_val(5);
    constexpr uint32_t v_bytes = 4096;  // FP32 tile

    cb_wait_front(cb_v, 1);
    const uint32_t v = get_read_ptr(cb_v);
    const uint32_t slot = get_write_ptr(cb_xchg) + index * v_bytes;  // same address on every helper
    for (uint32_t h = 0; h < helpers; ++h) {
        noc_async_write(
            v, get_noc_addr(get_arg_val<uint32_t>(1 + 2 * h), get_arg_val<uint32_t>(2 + 2 * h), slot), v_bytes);
    }
    noc_async_write_barrier();
    const uint32_t sem = get_semaphore(xchg_sem);
    for (uint32_t h = 0; h < helpers; ++h) {
        noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(1 + 2 * h), get_arg_val<uint32_t>(2 + 2 * h), sem), 1);
    }
    cb_pop_front(cb_v, 1);

    cb_wait_front(cb_x, PH);
    for (uint32_t i = 0; i < BND_DIRECT_N; ++i) {
        noc_semaphore_inc(
            get_noc_addr(
                get_arg_val<uint32_t>(BND_DIRECT_ARG + 2 * i),
                get_arg_val<uint32_t>(BND_DIRECT_ARG + 2 * i + 1),
                get_semaphore(0)),
            1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_x, PH);
}
