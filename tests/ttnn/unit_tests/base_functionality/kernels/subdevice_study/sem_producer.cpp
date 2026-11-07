// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Global-semaphore ping-pong producer (runs on sub-device 0). Iteration i: write a 2-word payload into the
// consumer's mailbox (with_payload = 1), raise the consumer's `ready` semaphore by one, then wait until the
// consumer has raised our `ack` semaphore to i. The round trip is timed on this core's wall clock.
// Stats page: [t_first, t_last, min_rtt, max_rtt, sum_rtt (64-bit), iterations, final ack].
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t cons_x = get_arg_val<uint32_t>(0);
    const uint32_t cons_y = get_arg_val<uint32_t>(1);
    const uint32_t ready_addr = get_arg_val<uint32_t>(2);
    const uint32_t ack_addr = get_arg_val<uint32_t>(3);
    const uint32_t mbox0_addr = get_arg_val<uint32_t>(4);
    const uint32_t mbox1_addr = get_arg_val<uint32_t>(5);
    const uint32_t n_iter = get_arg_val<uint32_t>(6);
    const uint32_t seed = get_arg_val<uint32_t>(7);
    const uint32_t stats_addr = get_arg_val<uint32_t>(8);
    constexpr uint32_t with_payload = get_named_compile_time_arg_val("with_payload");
    constexpr uint32_t cb_stats = get_named_compile_time_arg_val("cb_stats");
    constexpr auto st_args = TensorAccessorArgs<0>();
    const auto st = TensorAccessor(st_args, stats_addr);

    const uint64_t ready_noc = get_noc_addr(cons_x, cons_y, ready_addr);
    const uint64_t mbox0_noc = get_noc_addr(cons_x, cons_y, mbox0_addr);
    const uint64_t mbox1_noc = get_noc_addr(cons_x, cons_y, mbox1_addr);
    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_addr);

    uint32_t min_rtt = 0xFFFFFFFF;
    uint32_t max_rtt = 0;
    uint64_t sum_rtt = 0;
    const uint64_t t_first = get_timestamp();
    for (uint32_t i = 1; i <= n_iter; ++i) {
        const uint32_t t0 = get_timestamp_32b();  // 32-bit modular: the 64-bit read is not wrap-safe
        if constexpr (with_payload) {
            noc_inline_dw_write(mbox0_noc, i * 0x9E3779B1u + seed);
            noc_inline_dw_write(mbox1_noc, (i ^ seed) + 0x7F4A7C15u);
            noc_async_write_barrier();
        }
        noc_semaphore_inc(ready_noc, 1);
        noc_semaphore_wait_min(ack, i);
        const uint32_t rtt = get_timestamp_32b() - t0;
        min_rtt = rtt < min_rtt ? rtt : min_rtt;
        max_rtt = rtt > max_rtt ? rtt : max_rtt;
        sum_rtt += rtt;
    }
    const uint64_t t_last = get_timestamp();
    noc_async_atomic_barrier();

    cb_reserve_back(cb_stats, 1);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_stats));
    for (uint32_t i = 0; i < 16; ++i) {
        s[i] = 0;
    }
    s[0] = static_cast<uint32_t>(t_first);
    s[1] = static_cast<uint32_t>(t_first >> 32);
    s[2] = static_cast<uint32_t>(t_last);
    s[3] = static_cast<uint32_t>(t_last >> 32);
    s[4] = min_rtt;
    s[5] = max_rtt;
    s[6] = static_cast<uint32_t>(sum_rtt);
    s[7] = static_cast<uint32_t>(sum_rtt >> 32);
    s[8] = n_iter;
    s[9] = *ack;
    noc_async_write(get_write_ptr(cb_stats), st.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
