// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr bool capture_raw = get_compile_time_arg_val(0) != 0;
    constexpr auto sa = TensorAccessorArgs<1>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(sa, get_arg_val<uint32_t>(0), 4096);
    const auto output = TensorAccessor(oa, get_arg_val<uint32_t>(1), 512);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t count = get_arg_val<uint32_t>(4);
    const uint32_t cx = get_arg_val<uint32_t>(5);
    const uint32_t cy = get_arg_val<uint32_t>(6);
    const uint32_t part = get_arg_val<uint32_t>(7);
    const uint32_t scratch = get_write_ptr(10);
    const uint32_t mailbox = get_write_ptr(13);
    const uint64_t ready = get_noc_addr(cx, cy, get_semaphore(part));
    auto* consumed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(4));
    for (uint32_t item = 0; item < count; ++item) {
        const uint32_t head = (first + item * stride) / 4;
        if (item >= 2) {
            noc_semaphore_wait_min(consumed, item - 1);
        }
        cb_wait_front(7, 4);
        const uint32_t state_l1 = get_read_ptr(7);
        for (uint32_t kr = 0; kr < 4; ++kr) {
            noc_async_write_tile(head * 16 + kr * 4 + part, state, state_l1 + kr * 4096);
        }
        cb_wait_front(8, 1);
        const auto* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(8));
        auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
        for (uint32_t i = 0; i < 32; ++i) {
            row[i] = tile[(i / 16) * 256 + i % 16];
        }
        if constexpr (capture_raw) {
            noc_async_write(scratch, output.get_noc_addr(head, part * 128), 128);
        }
        noc_async_write(scratch, get_noc_addr(cx, cy, mailbox + (item % 2) * 512 + part * 128), 128);
        // Payload and canonical state writes complete before publication and
        // source reuse. Ready/credit counters are cumulative within a replay.
        noc_async_write_barrier();
        noc_semaphore_inc(ready, 1);
        cb_pop_front(7, 4);
        cb_pop_front(8, 1);
    }
    noc_async_atomic_barrier();
    noc_semaphore_wait_min(consumed, count);
}
