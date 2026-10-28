// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// One long-running kernel that records many NoC events, so the profiler's L1 marker ring fills and quick-pushes to
// DRAM many times inside a single launch (as a persistent kernel does). Each noc_async_read is one TS_DATA NoC event
// (4 words); the ring holds ~120 of them, so LOOP_COUNT=4000 gives ~33 mid-kernel pushes per RISC.
// The user zone spans every push, so its end lands in a different DRAM chunk than its start.
// Defines: LOOP_COUNT (reads), DELAY_NOPS (nops between reads, stretches the kernel), ITER_ZONE (1 = also open a
// zone around every read, as a model kernel with per-iteration zones does), PUSH_EVERY (n > 0 = every n-th
// iteration calls kernel_profiler::quick_push() directly, so a push can follow the ring-full push within a few
// dozen cycles), MCAST_EVERY (n > 0 = every n-th iteration also issues a linked multicast write to the whole core
// range, which makes the NoC tracer quick-push pre-emptively via quick_push_if_linked).
// Runtime args: page size, then the NoC x0 y0 x1 y1 of the multicast range (the test's core range).
void kernel_main() {
    const uint32_t page_size = get_arg_val<uint32_t>(0);
    const uint32_t l1_base = get_write_ptr(0);  // CB 0 has two pages: page 0 is read into page 1 over the local NoC
    const uint64_t src_noc_addr = get_noc_addr(l1_base);
#if MCAST_EVERY > 0
    const uint64_t mcast_noc_addr = get_noc_multicast_addr(
        get_arg_val<uint32_t>(1),
        get_arg_val<uint32_t>(2),
        get_arg_val<uint32_t>(3),
        get_arg_val<uint32_t>(4),
        l1_base + page_size);
    const uint32_t mcast_num_dests = get_arg_val<uint32_t>(5);  // receivers, the sender itself is not one
#endif
    {
        DeviceZoneScopedN("TEST-NOC-QUICK-PUSH");
        for (uint32_t i = 0; i < LOOP_COUNT; i++) {
#if ITER_ZONE
            DeviceZoneScopedN("TEST-ITER");
#endif
            noc_async_read(src_noc_addr, l1_base + page_size, page_size);
            noc_async_read_barrier();
#if PUSH_EVERY > 0 && defined(PROFILE_KERNEL)
            if (i % PUSH_EVERY == 0) {
                kernel_profiler::quick_push();
            }
#endif
#if MCAST_EVERY > 0
            if (i % MCAST_EVERY == 0) {
                // linked write, then the closing unlinked one (a linked chain must be terminated)
                noc_async_write_multicast(l1_base, mcast_noc_addr, page_size, mcast_num_dests, /*linked=*/true);
                noc_async_write_multicast(l1_base, mcast_noc_addr, page_size, mcast_num_dests, /*linked=*/false);
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
