// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert groups, weight injector writer (NCRISC, NOC1): lands each target core's slice of expert e (group g = e %
// G, its n = e / G-th expert) in slot n % 2 of the core's weight rings (gate/up ring, then down ring: same L1
// addresses on every core), once the core freed expert n - 2 (its credit word here >= n - 1), then increments the
// core's WARR semaphore (the fabric would fuse that increment into the last packet).
// RT: 0 injector index i, then per (group, target t): noc x, noc y. CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 CH, 5 NT,
// 6 NINJ, 7 G, 8 wgu ring addr, 9 wd ring addr, 10 credit words addr (16 B a (g, t))
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t SEM_WARR = 0;

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t CH = get_compile_time_arg_val(4), NT = get_compile_time_arg_val(5);
    constexpr uint32_t NINJ = get_compile_time_arg_val(6), G = get_compile_time_arg_val(7);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(8), wd_ring = get_compile_time_arg_val(9);
    constexpr uint32_t cred = get_compile_time_arg_val(10);
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS, WTOT = WGU + WDN;
    const uint32_t tb = get_tile_size(0);
    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t g = e % G, n = e / G;
        for (uint32_t t = 0; t < NT; ++t) {
            const uint32_t tx = get_arg_val<uint32_t>(1 + 2 * (g * NT + t)), ty = get_arg_val<uint32_t>(2 + 2 * (g * NT + t));
            if (n >= 2) {
                volatile tt_l1_ptr uint32_t* c =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred + (g * NT + t) * 16);
                do {
                    invalidate_l1_cache();
                } while (*c < n - 1);
            }
            for (uint32_t u0 = 0; u0 < WTOT; u0 += CH) {
                cb_wait_front(0, CH);
                const uint32_t dst = u0 < WGU ? wgu_ring + ((n & 1) * WGU + u0) * tb
                                              : wd_ring + ((n & 1) * WDN + u0 - WGU) * tb;
                noc_async_write(get_read_ptr(0), get_noc_addr(tx, ty, dst), CH * tb);
                noc_async_writes_flushed();
                cb_pop_front(0, CH);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(get_noc_addr(tx, ty, get_semaphore(SEM_WARR)), 1);
        }
    }
    noc_async_atomic_barrier();
}
