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
// iteration calls kernel_profiler::quick_push() twice back to back), MCAST_EVERY (n > 0 = every n-th iteration also issues a linked multicast write to the whole core
// range, which makes the NoC tracer quick-push pre-emptively via quick_push_if_linked).
// Runtime args: page size, then the NoC x0 y0 x1 y1 of the multicast range (the test's core range).
void kernel_main() {
    const uint32_t page_size = get_arg_val<uint32_t>(0);
    const uint32_t l1_base = get_write_ptr(0);  // CB 0 has two pages: page 0 is read into page 1 over the local NoC
    const uint64_t src_noc_addr = get_noc_addr(l1_base);
#if MCAST_EVERY > 0
    // Multicast to the OTHER column of the 2x2 core range (a sender inside its own multicast range hangs, jobs
    // 161289/161385): runtime args give the NoC x of both columns and the NoC y of both rows.
    const uint32_t col_a = get_arg_val<uint32_t>(1);
    const uint32_t row_a = get_arg_val<uint32_t>(2);
    const uint32_t col_b = get_arg_val<uint32_t>(3);
    const uint32_t row_b = get_arg_val<uint32_t>(4);
    const uint32_t other_col = (my_x[0] == col_a) ? col_b : col_a;
    // A multicast rectangle is given in the issuing NoC's traversal order: NOC0 start->end, NOC1 end->start
    // (DYNAMIC_NOC_X/Y flip the coordinates but do not swap the corners). NCRISC issues on NOC1.
    const uint32_t row_lo = row_a < row_b ? row_a : row_b;
    const uint32_t row_hi = row_a < row_b ? row_b : row_a;
    const uint64_t mcast_noc_addr =
        (noc_index == 0) ? get_noc_multicast_addr(other_col, row_lo, other_col, row_hi, l1_base + page_size)
                         : get_noc_multicast_addr(other_col, row_hi, other_col, row_lo, l1_base + page_size);
    constexpr uint32_t mcast_num_dests = 2;
#endif
    {
        DeviceZoneScopedN("TEST-NOC-QUICK-PUSH");
        for (uint32_t i = 0; i < LOOP_COUNT; i++) {
#if ITER_ZONE
            DeviceZoneScopedN("TEST-ITER");
#endif
#if MCAST_EVERY > 0 && defined(PROFILE_KERNEL)
            const uint32_t w_before = kernel_profiler::wIndex;
#endif
            noc_async_read(src_noc_addr, l1_base + page_size, page_size);
            noc_async_read_barrier();
#if PUSH_EVERY > 0 && defined(PROFILE_KERNEL)
            if (i % PUSH_EVERY == 0) {
                // Two pushes back to back: the first pushes the ring as it is (PUSH_EVERY events), the second, a few
                // dozen cycles later, pushes a 16-word chunk holding only the header and its own start/end pair.
                // That is the pair the broken captures show: a ring-full push followed ~75 cycles later by the
                // pre-emptive push of quick_push_if_linked.
                kernel_profiler::quick_push();
                kernel_profiler::quick_push();
            }
#endif
#if MCAST_EVERY > 0
            // MCAST_EVERY == 1: right after a read whose event filled the ring and quick-pushed it (wIndex dropped),
            // which is the model's pattern (a ring-full push, then ~75 cycles later the pre-emptive push of
            // quick_push_if_linked before a linked multicast). MCAST_EVERY > 1: every n-th iteration instead.
#if defined(PROFILE_KERNEL)
            const bool ring_just_pushed = kernel_profiler::wIndex < w_before;
#else
            const bool ring_just_pushed = false;
#endif
            if (MCAST_EVERY == 1 ? ring_just_pushed : (i % MCAST_EVERY == 0)) {
                // a linked chain of 4 writes, then the closing unlinked one (MCAST_WRITES = 5 NoC events)
                for (uint32_t k = 0; k < 4; k++) {
                    noc_async_write_multicast(l1_base, mcast_noc_addr, page_size, mcast_num_dests, /*linked=*/true);
                }
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
