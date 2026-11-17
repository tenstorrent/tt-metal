// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert 5 x 5 groups (ws_g5.py), weight injector writer (NCRISC): for every expert e (group g = e % G, its n-th
// expert, n = e / G) lands the target core's slice (weight set (e / G) % P, preloaded in CB 0) in slot n % 2 of the
// core's gate/up and down rings (same L1 addresses on every core), once the core freed expert n - 2 (credit word
// g >= n - 1), then increments the core's arrival word n % 2.
// RT: 0 WGU (gate/up tiles of the target's slice), 1 WDN, then per group: target noc x, y.
// CT: 0 NEXP, 1 G, 2 P, 3 WTOT_MAX, 4 CH, 5 WGU_MAX, 6 WDN_MAX, 7 wgu ring, 8 wd ring, 9 credits (16 B a group),
//     10 arrival words (16 B a slot)
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), G = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2), WTOT_MAX = get_compile_time_arg_val(3);
    constexpr uint32_t CH = get_compile_time_arg_val(4);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(7), wd_ring = get_compile_time_arg_val(8);
    constexpr uint32_t cred = get_compile_time_arg_val(9), warr = get_compile_time_arg_val(10);
    const uint32_t WGU = get_arg_val<uint32_t>(0), WDN = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(0);
    cb_wait_front(0, P * WTOT_MAX);
    const uint32_t pre = get_read_ptr(0);
    auto chunks = [&](uint32_t src, uint64_t dst_noc_base, uint32_t dst, uint32_t n) {
        for (uint32_t u = 0; u < n; u += CH) {
            const uint32_t m = n - u < CH ? n - u : CH;
            noc_async_write(src + u * tb, dst_noc_base | (dst + u * tb), m * tb);
            noc_async_writes_flushed();
        }
    };
    for (uint32_t e = 0; e < NEXP; ++e) {
        const uint32_t g = e % G, n = e / G;
        const uint32_t tx = get_arg_val<uint32_t>(2 + 2 * g), ty = get_arg_val<uint32_t>(3 + 2 * g);
        if (n >= 2) {
            volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred + g * 16);
            do {
                invalidate_l1_cache();
            } while (*c < n - 1);
        }
        const uint32_t src = pre + ((e / G) % P) * WTOT_MAX * tb;
        const uint64_t base = get_noc_addr(tx, ty, 0);
        chunks(src, base, wgu_ring + (n & 1) * WGU * tb, WGU);
        chunks(src + WGU * tb, base, wd_ring + (n & 1) * WDN * tb, WDN);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(tx, ty, warr + (n & 1) * 16), 1);
    }
    noc_async_atomic_barrier();
}
