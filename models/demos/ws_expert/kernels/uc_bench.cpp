// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Unicast microbench (BRISC on NOC0 or NCRISC on NOC1): N blocks of B bytes into a successor's ring of S slots (same
// L1 address on every core).
//   MODE 0 (flush): write, then noc_async_writes_flushed() after every block (as the lane chain's forwarders do)
//   MODE 1 (stream): writes back to back, one barrier at the end
//   MODE 2 (chain): the head sends block i (payload + 4-byte flag i + 1 from the same command buffer) once the tail has
//     credited block i - S; every other core forwards block i once its flag has landed (flushed after each, as the
//     forwarders do); the tail stamps each arrival and credits the head (inline write). A dedicated RISC per hop.
// Stamps: [0] start, [1] end, [64 + i] tail arrival of block i (i < 64), [128 + i] head send of block i (i < 64).
// CT: 0 N, 1 B, 2 S, 3 ring addr, 4 flags addr (16 B a slot), 5 credit addr, 6 stamps addr, 7 MODE
// RT: 0 successor x, 1 successor y (0xFFFF: none), 2 is head, 3 head x, 4 head y
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0), B = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2), ring = get_compile_time_arg_val(3);
    constexpr uint32_t flags = get_compile_time_arg_val(4), credit = get_compile_time_arg_val(5);
    constexpr uint32_t MODE = get_compile_time_arg_val(7);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(6));
    const uint32_t sx = get_arg_val<uint32_t>(0), sy = get_arg_val<uint32_t>(1);
    const bool head = get_arg_val<uint32_t>(2) != 0;
    const uint32_t hx = get_arg_val<uint32_t>(3), hy = get_arg_val<uint32_t>(4);
    const bool succ = sy != 0xFFFF;
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    auto word = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    st[0] = wall();
    if constexpr (MODE < 2) {
        if (succ) {
            for (uint32_t i = 0; i < N; ++i) {
                noc_async_write(ring + (i % S) * B, get_noc_addr(sx, sy, ring + (i % S) * B), B);
                if constexpr (MODE == 0) {
                    noc_async_writes_flushed();
                }
            }
            noc_async_write_barrier();
        }
    } else {
        for (uint32_t i = 0; i < N; ++i) {
            const uint32_t s = i % S, f = flags + s * 16;
            if (head) {
                if (i >= S) {
                    while (*word(credit) < i - S + 1) {
                        invalidate_l1_cache();
                    }
                }
                *word(f) = i + 1;
            } else {
                while (*word(f) < i + 1) {
                    invalidate_l1_cache();
                }
            }
            if (succ) {
                noc_async_write(ring + s * B, get_noc_addr(sx, sy, ring + s * B), B);
                noc_async_write(f, get_noc_addr(sx, sy, f), 4);
                noc_async_writes_flushed();
                if (head && i < 64) {
                    st[128 + i] = wall();
                }
            } else {
                if (i < 64) {
                    st[64 + i] = wall();
                }
                noc_inline_dw_write(get_noc_addr(hx, hy, credit), i + 1);
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
    }
    st[1] = wall();
    // NIU state of this RISC's NoC: [200] NIU_CFG_0 (bit 16: NIU request FIFOs on), [201] CMD_BUF_AVAIL, [202]
    // CMD_BUF_OVFL (sticky: a command written while the FIFO was full was dropped)
    auto niu = [](uint32_t a) { return *reinterpret_cast<volatile uint32_t*>(a + noc_index * NOC_INSTANCE_OFFSET); };
    st[200] = niu(NOC_CFG(NIU_CFG_0));
    st[201] = niu(CMD_BUF_AVAIL);
    st[202] = niu(CMD_BUF_OVFL);
}
