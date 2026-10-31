// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Router logits, K-split, writer. Chunk kc > 0 of column n writes its Mt fp32 partial tiles into slot kc - 1 of the
// column reducer's cb_recv (the same L1 address on every core) and bumps the reducer's semaphore; the reducer (kc 0)
// waits for the KC - 1 partials, hands them to compute and writes the Mt score tiles (m, n) of out [Mt * 32, Nt * 32].
// Common args: out address, then the Nt reducers' NOC (x, y).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t KC = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr uint32_t cb_part = 2, cb_recv = 3, cb_out = 16, tb = 4096;
    const uint32_t c = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t n = c / KC, kc = c % KC;
    const uint32_t sem = get_semaphore(0);
    if (kc > 0) {
        const uint32_t rx = get_common_arg_val<uint32_t>(1 + 2 * n), ry = get_common_arg_val<uint32_t>(2 + 2 * n);
        cb_wait_front(cb_part, Mt);
        const uint32_t dst = get_write_ptr(cb_recv) + (kc - 1) * Mt * tb;
        noc_async_write(get_read_ptr(cb_part), get_noc_addr(rx, ry, dst), Mt * tb);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(rx, ry, sem), 1);
        noc_async_atomic_barrier();
        cb_pop_front(cb_part, Mt);
        return;
    }
    cb_reserve_back(cb_recv, (KC - 1) * Mt);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem);
    noc_semaphore_wait(s, KC - 1);
    noc_semaphore_set(s, 0);
    cb_push_back(cb_recv, (KC - 1) * Mt);
    const auto o = TensorAccessor(o_args, get_common_arg_val<uint32_t>(0), tb);
    cb_wait_front(cb_out, Mt);
    for (uint32_t m = 0; m < Mt; ++m) {
        noc_async_write_tile(m * Nt + n, o, get_read_ptr(cb_out) + m * tb);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, Mt);
}
