// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast microbench (BRISC or NCRISC): each sender issues N multicasts of B bytes from its L1 into a destination
// rectangle (same L1 address on every destination), optionally followed by a 4-byte flag multicast each, with a write
// barrier after each multicast (BARRIER 1) or only at the end (BARRIER 0). Stamps: [0] start, [1] end.
// CT: 0 N, 1 B, 2 x0, 3 y0, 4 x1, 5 y1, 6 ndest, 7 src addr, 8 dst addr, 9 flag addr, 10 FLAG, 11 BARRIER, 12 stamps
#include "api/dataflow/dataflow_api.h"
#ifndef LINKED
#define LINKED false
#endif

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t x0 = get_compile_time_arg_val(2), y0 = get_compile_time_arg_val(3);
    constexpr uint32_t x1 = get_compile_time_arg_val(4), y1 = get_compile_time_arg_val(5);
    constexpr uint32_t nd = get_compile_time_arg_val(6);
    constexpr uint32_t src = get_compile_time_arg_val(7);
    constexpr uint32_t dst = get_compile_time_arg_val(8);
    constexpr uint32_t flag = get_compile_time_arg_val(9);
    constexpr uint32_t FLAG = get_compile_time_arg_val(10);
    constexpr uint32_t BARRIER = get_compile_time_arg_val(11);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(12));
    const uint32_t slot = get_arg_val<uint32_t>(0);  // this sender's destination offset (bytes)
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    st[0] = wall();
    for (uint32_t i = 0; i < N; ++i) {
#ifdef LOOPBACK
        noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(x0, y0, x1, y1, dst + slot), B, nd, LINKED);
#ifdef LFLAG
        noc_semaphore_set_multicast_loopback_src(
            src, get_noc_multicast_addr(x0, y0, x1, y1, flag + (slot / 16 % 64) * 16), nd);
        noc_async_writes_flushed();
#endif
#else
        noc_async_write_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, dst + slot), B, nd);
#endif
        if constexpr (BARRIER) {
            noc_async_write_barrier();
        }
        if constexpr (FLAG) {
            noc_semaphore_set_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, flag + (slot / 16 % 64) * 16), nd);
            if constexpr (BARRIER) {
                noc_async_write_barrier();
            }
        }
    }
    noc_async_write_barrier();
    st[1] = wall();
}
