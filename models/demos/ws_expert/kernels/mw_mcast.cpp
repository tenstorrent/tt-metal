// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast wedge repro, multicaster (BRISC, NOC0): blocks base .. base + ITER - 1 of PB bytes from one staging buffer
// into ring slot s % NX of every core of a rectangle (loopback, unlinked), each followed by its 4-byte flag (s + 1) from
// the same command buffer, then a flush: the lane design's x multicast (wl_writer QX). Slot s % NX is reused only once
// every receiver acknowledged block s - NX (prog word c, written by receiver c's NCRISC). SELF_RD keeps this RISC's own
// DRAM reads in flight between multicasts, as the lane BRISC does: bit 0 gather-like batches (32 reads of 1280 B),
// bit 1 own-expert-like reads (4 reads of 13824 B in flight, transaction ids 1..4). At the end it waits for every
// acknowledgement, then writes stopv into the stop word of every background core. MW_NO_MCAST (a control): no
// multicast, only the own reads for about a multicast's time a turn. MW_MC_BARRIER: a write barrier (every
// acknowledgement) after each multicast instead of the flush.
// CT: 0 ITER, 1 PB, 2 NX, 3-6 rectangle (NOC0, start corner first), 7 GC (destinations, this core included), 8 ring,
//     9 flags, 10 prog, 11 staging, 12 flag source word, 13 stamps, 14 stop word, 15 SELF_RD, 16 landing (64 B
//     aligned), 17 DRAM pages, then TensorAccessorArgs(DRAM buffer)
// RT: 0 base, 1 stopv, 2 DRAM buffer address, 3 background cores, then their (x, y) NOC0 pairs
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE void mcast_with_flag(
    uint32_t src, uint64_t dst, uint32_t bytes, uint32_t n, uint32_t flag_src, uint64_t flag_dst) {
    noc_async_write_multicast_loopback_src(src, dst, bytes, n, false);
    noc_async_write_multicast_loopback_src(flag_src, flag_dst, 4, n, false);
}

void kernel_main() {
    constexpr uint32_t ITER = get_compile_time_arg_val(0);
    constexpr uint32_t PB = get_compile_time_arg_val(1);
    constexpr uint32_t NX = get_compile_time_arg_val(2);
    constexpr uint32_t x0 = get_compile_time_arg_val(3), y0 = get_compile_time_arg_val(4);
    constexpr uint32_t x1 = get_compile_time_arg_val(5), y1 = get_compile_time_arg_val(6);
    constexpr uint32_t GC = get_compile_time_arg_val(7);
    constexpr uint32_t ring = get_compile_time_arg_val(8), flags = get_compile_time_arg_val(9);
    constexpr uint32_t prog = get_compile_time_arg_val(10), src = get_compile_time_arg_val(11);
    constexpr uint32_t scr = get_compile_time_arg_val(12);
    constexpr uint32_t SELF_RD = get_compile_time_arg_val(15);
    constexpr uint32_t land = get_compile_time_arg_val(16), NPAGES = get_compile_time_arg_val(17);
    constexpr auto dargs = TensorAccessorArgs<18>();
    constexpr uint32_t PAGE = 16384, GB = 1280, OB = 13824, D = 4;
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(13));
    volatile tt_l1_ptr uint32_t* pg = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(prog);
    volatile tt_l1_ptr uint32_t* sw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr);
    const uint32_t base = get_arg_val<uint32_t>(0), stopv = get_arg_val<uint32_t>(1);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(2), PAGE);
    const uint32_t nbg = get_arg_val<uint32_t>(3);

    uint32_t g_seq = 0, o_issued = 0, o_done = 0;
    bool g_out = false;
    auto self_reads = [&]() {
        if constexpr (SELF_RD & 1) {
            if (g_out && ncrisc_noc_reads_flushed(noc_index)) {
                g_out = false;
            }
            if (!g_out) {
                for (uint32_t r = 0; r < 32; ++r) {
                    noc_async_read(ds.get_noc_addr((g_seq * 37 + r * 13) % NPAGES), land + r * GB, GB);
                }
                ++g_seq;
                g_out = true;
            }
        }
        if constexpr (SELF_RD & 2) {
            if (o_done < o_issued && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 1 + o_done % D)) {
                ++o_done;
            }
            if (o_issued - o_done < D) {
                noc_async_read_set_trid(1 + o_issued % D);
                noc_async_read(ds.get_noc_addr((o_issued * 7 + 3) % NPAGES), land + 32 * GB + (o_issued % D) * OB, OB);
                noc_async_read_set_trid(0);
                ++o_issued;
            }
        }
    };
    auto wait_acks = [&](uint32_t target) {
        for (uint32_t c = 0; c < GC; ++c) {
            while (true) {
                invalidate_l1_cache();
                if (pg[c * 4] >= target) {
                    break;
                }
                self_reads();
            }
        }
    };
#ifdef MW_NO_MCAST
    // (control: no multicast; the same loop of own reads, about a multicast's time per turn)
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    for (uint32_t i = 0; i < ITER; ++i) {
        const uint32_t t = wall();
        while (wall() - t < 1200) {
            self_reads();
        }
        st[0] = base + i + 1;
    }
#else
    for (uint32_t i = 0; i < ITER; ++i) {
        const uint32_t s = base + i, slot = s % NX;
        if (s >= NX) {
            wait_acks(s - NX + 1);
        }
        *sw = s + 1;
        mcast_with_flag(
            src,
            get_noc_multicast_addr(x0, y0, x1, y1, ring + slot * PB),
            PB,
            GC,
            scr,
            get_noc_multicast_addr(x0, y0, x1, y1, flags + slot * 16));
#ifdef MW_MC_BARRIER
        noc_async_write_barrier();
#else
        noc_async_writes_flushed();
#endif
        self_reads();
        st[0] = s + 1;
    }
    wait_acks(base + ITER);
#endif
    for (uint32_t b = 0; b < nbg; ++b) {
        noc_inline_dw_write(
            get_noc_addr(get_arg_val<uint32_t>(4 + 2 * b), get_arg_val<uint32_t>(5 + 2 * b), get_compile_time_arg_val(14)),
            stopv);
    }
    noc_async_read_barrier();
    noc_async_write_barrier();
    st[1] = stopv;
}
