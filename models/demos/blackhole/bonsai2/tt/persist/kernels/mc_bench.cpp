// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Handoff microbenchmark (persistent decode groundwork): cores 0 .. S-1 each multicast a 2-KB block into slot s of
// every core's L1 buffer and then set flag s (16 B apart); every core waits for all S flags and clears them.
// Args (common): buf (L1 address), flag_off, S, nrect, then per rectangle x0, y0, x1, y1, cores; grid_x.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "../../ternary/kernels/core_span.hpp"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t buf = get_common_arg_val<uint32_t>(0);
    const uint32_t flag_off = get_common_arg_val<uint32_t>(1);
    const uint32_t S = get_common_arg_val<uint32_t>(2);
    const uint32_t nrect = get_common_arg_val<uint32_t>(3);
    const uint32_t grid_x = get_common_arg_val<uint32_t>(4 + 5 * nrect);
    const uint32_t me = core_index(grid_x);
    const uint32_t one = get_semaphore(0);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(one) = 1;
#ifdef MC_HUB
    // hub variant: senders unicast their block into the hub's (core S, rectangle args hub_x, hub_y) buffer and bump
    // its semaphore 1; the hub then multicasts all S blocks and one flag (slot 0) once
    const uint32_t hub_x = get_common_arg_val<uint32_t>(5 + 5 * nrect),
                   hub_y = get_common_arg_val<uint32_t>(6 + 5 * nrect);
    const uint32_t cnt = get_semaphore(1);
    if (me < S) {
        noc_async_write(buf + me * 2048, get_noc_addr(hub_x, hub_y, buf + me * 2048), 2048);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, cnt), 1);
        noc_async_atomic_barrier();
    }
    if (me == S) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt), S);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt), 0);
        for (uint32_t q = 0; q < nrect; ++q) {
            const uint32_t x0 = get_common_arg_val<uint32_t>(4 + 5 * q), y0 = get_common_arg_val<uint32_t>(5 + 5 * q);
            const uint32_t x1 = get_common_arg_val<uint32_t>(6 + 5 * q), y1 = get_common_arg_val<uint32_t>(7 + 5 * q);
            const uint32_t nd = get_common_arg_val<uint32_t>(8 + 5 * q);
            const uint64_t d = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, buf)
                                              : get_noc_multicast_addr(x1, y1, x0, y0, buf);
            const uint64_t f = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, buf + flag_off)
                                              : get_noc_multicast_addr(x1, y1, x0, y0, buf + flag_off);
            noc_async_write_multicast_loopback_src(buf, d, S * 2048, nd);
            noc_semaphore_set_multicast_loopback_src(one, f, nd);
        }
        noc_async_write_barrier();
    }
    {
        volatile tt_l1_ptr uint32_t* fl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + flag_off);
        noc_semaphore_wait(fl, 1);
        noc_semaphore_set(fl, 0);
    }
#else
    {
        DeviceZoneScopedN("MC-send");
        if (me < S) {
            for (uint32_t q = 0; q < nrect; ++q) {
                const uint32_t x0 = get_common_arg_val<uint32_t>(4 + 5 * q),
                               y0 = get_common_arg_val<uint32_t>(5 + 5 * q);
                const uint32_t x1 = get_common_arg_val<uint32_t>(6 + 5 * q),
                               y1 = get_common_arg_val<uint32_t>(7 + 5 * q);
                const uint32_t nd = get_common_arg_val<uint32_t>(8 + 5 * q);
                // the senders are inside the rectangles: loopback (nd counts this core too)
                const uint64_t d = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, buf + me * 2048)
                                                  : get_noc_multicast_addr(x1, y1, x0, y0, buf + me * 2048);
                const uint64_t f = noc_index == 0 ? get_noc_multicast_addr(x0, y0, x1, y1, buf + flag_off + 16 * me)
                                                  : get_noc_multicast_addr(x1, y1, x0, y0, buf + flag_off + 16 * me);
                noc_async_write_multicast_loopback_src(buf + me * 2048, d, 2048, nd);
                noc_semaphore_set_multicast_loopback_src(one, f, nd);
            }
            noc_async_write_barrier();
        }
    }
    {
        DeviceZoneScopedN("MC-wait");
        for (uint32_t s = 0; s < S; ++s) {
            volatile tt_l1_ptr uint32_t* fl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + flag_off + 16 * s);
            noc_semaphore_wait(fl, 1);
            noc_semaphore_set(fl, 0);
        }
    }
#endif
}
