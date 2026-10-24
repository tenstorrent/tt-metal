// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
//
// SPDX-License-Identifier: Apache-2.0

// Link stress: rounds x (notify -> responder -> wait -> read the reply line and check reply[i] == req * 64 + i).
// Result line at DIAG line 0: rounds_done, stale, timeouts, wall ticks total, worst round ticks, first bad req.
// Runtime args: l2_x, l2_y, base_hi, base_lo, timeout_us (per round), rounds. Stops at the first timeout.
// CB 0: >= 2 KiB.
#include "api/dataflow/dataflow_api.h"
#include "l2cpu_noc.h"

void kernel_main() {
    l2cpu_link c{
        noc_index,
        get_arg_val<uint32_t>(0),
        get_arg_val<uint32_t>(1),
        ((uint64_t)get_arg_val<uint32_t>(2) << 32) | get_arg_val<uint32_t>(3),
        get_write_ptr(0)};
    uint32_t timeout_us = get_arg_val<uint32_t>(4), rounds = get_arg_val<uint32_t>(5);
    uint32_t rl = c.l1 + 1024;
    volatile tt_l1_ptr uint32_t* rep = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rl);
    uint32_t stale = 0, timeouts = 0, first_bad = 0, tmax = 0, k = 0;
    uint32_t t0 = get_timestamp_32b();
    for (k = 0; k < rounds; k++) {
        uint32_t ts = get_timestamp_32b();
        uint32_t r = l2cpu_link_notify(c);
        if (!l2cpu_link_wait(c, timeout_us)) {
            timeouts++;
            break;
        }
        l2cpu_noc_read(c.noc, rl, c.x, c.y, c.base + L2CPU_LINK_OFF_REPLY, 64);
        noc_async_read_barrier();
        for (uint32_t i = 0; i < L2CPU_LINK_REPLY_WORDS; i++) {
            if (rep[i] != r * 64 + i) {
                if (!stale++) {
                    first_bad = r;
                }
                break;
            }
        }
        uint32_t dt = get_timestamp_32b() - ts;
        if (dt > tmax) {
            tmax = dt;
        }
    }
    uint32_t tt = get_timestamp_32b() - t0;
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(c.l1 + 64 * 4);
    for (uint32_t i = 0; i < 16; i++) {
        p[i] = 0;
    }
    p[0] = k;
    p[1] = stale;
    p[2] = timeouts;
    p[3] = tt;
    p[4] = tmax;
    p[5] = first_bad;
    l2cpu_noc_write(c.noc, c.l1 + 64 * 4, c.x, c.y, c.base + L2CPU_LINK_OFF_DIAG, 64);
    noc_async_write_barrier();
}
