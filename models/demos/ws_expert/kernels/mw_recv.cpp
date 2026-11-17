// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast wedge repro, receiver (NCRISC, NOC1), on every core of the rectangle: for blocks base .. base + ITER - 1,
// waits for the flag of ring slot s % NX to reach s + 1, then acknowledges the block with an inline write of s + 1 into
// its prog word on the multicaster (the lane design's x ring credits, wl_reader QB).
// CT: 0 ITER, 1 NX, 2 flags, 3 prog, 4 stamps
// RT: 0 base, 1 this core's index c, 2 multicaster x, 3 multicaster y
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t ITER = get_compile_time_arg_val(0);
    constexpr uint32_t NX = get_compile_time_arg_val(1);
    constexpr uint32_t flags = get_compile_time_arg_val(2), prog = get_compile_time_arg_val(3);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(4));
    volatile tt_l1_ptr uint32_t* fl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flags);
    const uint32_t base = get_arg_val<uint32_t>(0), c = get_arg_val<uint32_t>(1);
    const uint64_t ack = get_noc_addr(get_arg_val<uint32_t>(2), get_arg_val<uint32_t>(3), prog + c * 16);
    for (uint32_t i = 0; i < ITER; ++i) {
        const uint32_t s = base + i;
        while (true) {
            invalidate_l1_cache();
            if (fl[(s % NX) * 4] >= s + 1) {
                break;
            }
        }
        noc_inline_dw_write(ack, s + 1);
        st[4] = s + 1;
    }
    noc_async_write_barrier();
}
