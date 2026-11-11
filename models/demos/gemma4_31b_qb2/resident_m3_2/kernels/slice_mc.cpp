// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident slice, second h multicaster (NCRISC, NOC 1; M2b F2). slice_dm (BRISC, NOC 0) multicasts each round's
// h tiles into the first rectangles of the streamer cover; this kernel takes the remaining rectangles, so the two
// NoCs carry the round's h at once. Per h (seed, then rounds 1 .. rounds): wait for the BRISC's go count (it has
// cb_hout's front: the compute's h of this round, at cb_hout's base, published once in sem_src), multicast the slice's
// h into this kernel's rectangles, then +1 on their sem_slots, then report done. The BRISC pops cb_hout only after
// both halves are out, so cb_hout keeps one consumer (the BRISC). This RISC issues NoC transactions on NOC 1 only.
//
// Compile-time args: 0 rounds, 1 slice_bytes, 2 k, 3 sem_go, 4 sem_done, 5 sem_src, 6 sem_haddr, 7 sem_slots
// Runtime args: 0 rectangle count, then per rectangle physical x0, y0, x1, y1, cores

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t rounds = get_compile_time_arg_val(0);
constexpr uint32_t slice_bytes = get_compile_time_arg_val(1);
constexpr uint32_t k = get_compile_time_arg_val(2);
constexpr uint32_t sem_go = get_compile_time_arg_val(3);
constexpr uint32_t sem_done = get_compile_time_arg_val(4);
constexpr uint32_t sem_src = get_compile_time_arg_val(5);
constexpr uint32_t sem_haddr = get_compile_time_arg_val(6);
constexpr uint32_t sem_slots = get_compile_time_arg_val(7);

void kernel_main() {
    const uint32_t rects = get_arg_val<uint32_t>(0);
    auto rect = [&](uint32_t q, uint32_t i) { return get_arg_val<uint32_t>(1 + 5 * q + i); };
    // NOC 1 multicasts run high -> low on Blackhole: the rectangle's end corner goes first (as tt-metal's NOC_1
    // multicast senders swap start and end)
    auto mc = [&](uint32_t q, uint32_t addr) { return get_noc_multicast_addr(rect(q, 2), rect(q, 3), rect(q, 0), rect(q, 1), addr); };
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go));
    volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_done));
    volatile tt_l1_ptr uint32_t* srcw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_src));
    volatile tt_l1_ptr uint32_t* haddr_box = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_haddr));
    uint32_t h_dst = 0;
    for (uint32_t r = 0; r <= rounds; r++) {
        do {
            invalidate_l1_cache();
        } while (*go < r + 1);
        while (h_dst == 0) {
            invalidate_l1_cache();
            h_dst = *haddr_box;
        }
        const uint32_t src = *srcw;
        const uint32_t dst = h_dst + k * slice_bytes;
        for (uint32_t q = 0; q < rects; q++) {
            noc_async_write_multicast(src, mc(q, dst), slice_bytes, rect(q, 4));
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_inc_multicast(mc(q, get_semaphore(sem_slots)), 1, rect(q, 4));
        }
        noc_async_atomic_barrier();
        *done = r + 1;
    }
}
