// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Global-semaphore ping-pong consumer (runs on sub-device 1). Iteration i: wait until `ready` >= i, log
// [ready, mailbox0, mailbox1] as observed, then raise the producer's `ack` semaphore by one. The log
// (4 words per iteration) goes to page 0 of the output tensor so the host can check order and payload.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t prod_x = get_arg_val<uint32_t>(0);
    const uint32_t prod_y = get_arg_val<uint32_t>(1);
    const uint32_t ready_addr = get_arg_val<uint32_t>(2);
    const uint32_t ack_addr = get_arg_val<uint32_t>(3);
    const uint32_t mbox0_addr = get_arg_val<uint32_t>(4);
    const uint32_t mbox1_addr = get_arg_val<uint32_t>(5);
    const uint32_t n_iter = get_arg_val<uint32_t>(6);
    const uint32_t log_addr = get_arg_val<uint32_t>(7);
    constexpr uint32_t cb_log = get_named_compile_time_arg_val("cb_log");
    constexpr auto log_args = TensorAccessorArgs<0>();
    const auto log_out = TensorAccessor(log_args, log_addr);

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
    volatile tt_l1_ptr uint32_t* mbox0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox0_addr);
    volatile tt_l1_ptr uint32_t* mbox1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mbox1_addr);
    const uint64_t ack_noc = get_noc_addr(prod_x, prod_y, ack_addr);

    cb_reserve_back(cb_log, 1);
    volatile tt_l1_ptr uint32_t* log = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_log));
    for (uint32_t i = 1; i <= n_iter; ++i) {
        noc_semaphore_wait_min(ready, i);
        invalidate_l1_cache();
        log[4 * (i - 1) + 0] = *ready;
        log[4 * (i - 1) + 1] = *mbox0;
        log[4 * (i - 1) + 2] = *mbox1;
        log[4 * (i - 1) + 3] = 0;
        noc_semaphore_inc(ack_noc, 1);
    }
    noc_async_atomic_barrier();
    noc_async_write(get_write_ptr(cb_log), log_out.get_noc_addr(0), 16 * n_iter);
    noc_async_write_barrier();
}
