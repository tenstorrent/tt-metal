// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// One long-running kernel that records many NoC events, so the profiler's L1 marker ring fills and quick-pushes to
// DRAM many times inside a single launch (as a persistent kernel does). Each noc_async_read is one TS_DATA NoC event
// (4 words); the ring holds ~120 of them, so LOOP_COUNT=4000 gives ~33 mid-kernel pushes per RISC.
// The user zone spans every push, so its end lands in a different DRAM chunk than its start.
// Defines: LOOP_COUNT (reads), DELAY_NOPS (nops between reads, stretches the kernel), ITER_ZONE (1 = also open a
// zone around every read, as a model kernel with per-iteration zones does), MCAST_EVERY (n > 0 = every n-th
// iteration also issues a linked multicast write to itself, which makes the NoC tracer quick-push pre-emptively
// via quick_push_if_linked, at any ring fill level, so pushes can follow each other within a few dozen cycles).
void kernel_main() {
    const uint32_t page_size = get_arg_val<uint32_t>(0);
    const uint32_t l1_base = get_write_ptr(0);  // CB 0 has two pages: page 0 is read into page 1 over the local NoC
    const uint64_t src_noc_addr = get_noc_addr(l1_base);
#if MCAST_EVERY > 0
    // 1x1 multicast range = this core; loopback_src delivers it to the sender itself.
    const uint64_t mcast_noc_addr = get_noc_multicast_addr(my_x[0], my_y[0], my_x[0], my_y[0], l1_base + page_size);
#endif
    {
        DeviceZoneScopedN("TEST-NOC-QUICK-PUSH");
        for (uint32_t i = 0; i < LOOP_COUNT; i++) {
#if ITER_ZONE
            DeviceZoneScopedN("TEST-ITER");
#endif
            noc_async_read(src_noc_addr, l1_base + page_size, page_size);
            noc_async_read_barrier();
#if MCAST_EVERY > 0
            if (i % MCAST_EVERY == 0) {
                // linked write, then the closing unlinked one (a linked chain must be terminated)
                noc_async_write_multicast_loopback_src(l1_base, mcast_noc_addr, page_size, 1, /*linked=*/true);
                noc_async_write_multicast_loopback_src(l1_base, mcast_noc_addr, page_size, 1, /*linked=*/false);
                noc_async_write_barrier();
            }
#endif
#if DELAY_NOPS > 0
            for (uint32_t d = 0; d < DELAY_NOPS; d++) {
                asm volatile("nop");
            }
#endif
        }
    }
}
