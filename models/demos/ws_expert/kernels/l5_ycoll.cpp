// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert 5 x 5 lane (ws_lane5.py, L5Cfg.yc): a group's row collector, on a spare core in the column west of the
// group (outside its rectangle). The group's 25 cores own column slices of every down row tile; each sends its rows of
// row tile n (n % NCY == this collector) into staging slot k % NSL (k = n / NCY; 32 rows of ROWB bytes at STAGE), then
// + 1 on the slot's arrival word. Once all 25 arrived, the collector writes each row whole (one ROWB write) to its
// destination row of the row-major result buffer (the expert's destination list), posted, and once the writes have
// left returns the slot: k + 1 into word c of CRED on all 25 cores. The cores need many small writes either way; the
// collector turns each row's 25 pieces into one DRAM write.
// Everything runs on the NCRISC, NOC1 (the destination list reads, the row writes, the slot credits into the rectangle).
// RT 10 (split, L5Cfg.yc_n0): the row writes on the BRISC instead (this file with YC_WRITER, NOC0), for a collector
// whose side's DRAM column lies east of it outside every group rectangle; the two RISCs hand over through GO / NRW /
// SENT (L1 words). At the end the row writes are fenced (an
// acknowledged 4-B write into every DRAM bank's spare row, same NoC and VC) and the group's core 0 gets + 1 on word 3
// of CRED. CT 12 (SINK, a diagnostic): no row writes (the cores' traffic and the protocol only).
// YC_OEAST (L5Cfg.oeast): row d at page (d >> 2) 8 + 4 + (d & 3), every page in banks 4-7 (the east DRAM column); the
// fence covers those 4 banks only (fence row: a multiple of 8). Stamp 21 counts row writes whose page is not at NoC x 9.
// CT: 0 NSL, 1 ROWB, 2 STAGE, 3 YW (words: arrival 16 s, GO 64, NRW 80, SENT 96, FDONE 112, destinations 128, fence
//     source 256), 4 CRED (on the group cores), 5 RMAX, 6 NEXP, 7 table page bytes, 8 counts L1, 9 table L1, 10 GCN,
//     11 (unused), 12 SINK, then TensorAccessorArgs(counts), (table), (dest), (out).
// RT: 0 g, 1 c, 2 NCY, 3 experts of the group, 4 counts addr, 5 table addr, 6 dest addr, 7 out addr, 8 fence row,
//     9 stamps addr, 10 split, 11 .. 11 + GCN - 1 the group cores' NoC coordinates (x | y << 8; core 0 first).
// Stamps (words): 0 start, 1 total, 2 arrival waits, 3 destination waits, 4 row writes (issue to sent; split: GO to
//     SENT), 5 credits, 6 row tiles, 7 end (wall clock), 8 the end's fence, 9 / 10 the row writes' issue / sent waits
//     (not split); the BRISC: 16 GO waits, 17 issue, 18 sent waits, 19 fence, 20 row tiles.
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t NSL = get_compile_time_arg_val(0), ROWB = get_compile_time_arg_val(1);
    constexpr uint32_t STAGE = get_compile_time_arg_val(2), YW = get_compile_time_arg_val(3);
    constexpr uint32_t CRED = get_compile_time_arg_val(4), RMAX = get_compile_time_arg_val(5);
    constexpr uint32_t NEXP = get_compile_time_arg_val(6), TB = get_compile_time_arg_val(7);
    constexpr uint32_t CNT = get_compile_time_arg_val(8), TAB = get_compile_time_arg_val(9);
    constexpr uint32_t GCN = get_compile_time_arg_val(10);
    constexpr bool SINK = get_compile_time_arg_val(12) != 0;
    constexpr auto cargs = TensorAccessorArgs<13>();
    constexpr auto targs = TensorAccessorArgs<cargs.next_compile_time_args_offset()>();
    constexpr auto dargs = TensorAccessorArgs<targs.next_compile_time_args_offset()>();
    constexpr auto oargs = TensorAccessorArgs<dargs.next_compile_time_args_offset()>();
    constexpr uint32_t SLOTB = 32 * ROWB;
    constexpr uint32_t GO = YW + 64, NRW = YW + 80, SENT = YW + 96, FDONE = YW + 112, DBUF = YW + 128, FSRC = YW + 256;
    constexpr uint32_t END = 0xFFFFFFFFu;
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };

    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(7), ROWB);
    const uint32_t fence_row = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint32_t* st = W(get_arg_val<uint32_t>(9));
    const uint32_t t_start = wall();

#if defined(YC_OEAST)
    auto orow = [](uint32_t d) { return (d >> 2) * 8 + 4 + (d & 3); };
    constexpr uint32_t FB0 = 4;
#else
    auto orow = [](uint32_t d) { return d; };
    constexpr uint32_t FB0 = 0;
#endif
    uint32_t off_east = 0;
    // the rows of one row tile from staging slot s to their destinations (posted), then wait until they have left
    auto write_rows = [&](uint32_t s, uint32_t nr, uint32_t& t_issue, uint32_t& t_sent) {
        const uint32_t a = wall();
        volatile tt_l1_ptr uint32_t* dst = W(DBUF);
        for (uint32_t r = 0; r < nr; ++r) {
            const uint64_t na = os.get_noc_addr(orow(dst[r]));
#if defined(YC_OEAST)
            off_east += ((na >> NOC_ADDR_LOCAL_BITS) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1)) != 9;
#endif
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(STAGE + s * SLOTB + r * ROWB, na, ROWB);
        }
        const uint32_t b = wall();
        while (!ncrisc_noc_posted_writes_sent(noc_index)) {
        }
        t_issue += b - a;
        t_sent += wall() - b;
    };
    auto fence = [&]() {
        noc_async_posted_writes_flushed();
        for (uint32_t i = FB0; i < NUM_DRAM_BANKS; ++i) {
            noc_async_write(FSRC, os.get_noc_addr(fence_row + i), 4);
        }
        noc_async_write_barrier();
    };

#if defined(YC_WRITER)
    // ---- the BRISC's row writes (NOC0), handed over by the NCRISC's control
    uint32_t k = 0, t_go = 0, t_issue = 0, t_sent = 0;
    for (;;) {
        const uint32_t a = wall();
        uint32_t go;
        do {
            invalidate_l1_cache();
            go = *W(GO);
        } while (go == k);
        t_go += wall() - a;
        if (go == END) {
            break;
        }
        const uint32_t v = *W(NRW);
        write_rows(v >> 8, v & 0xFF, t_issue, t_sent);
        ++k;
        *W(SENT) = k;
    }
    const uint32_t f = wall();
    fence();
    st[16] = t_go;
    st[17] = t_issue;
    st[18] = t_sent;
    st[19] = wall() - f;
    st[20] = k;
    st[21] = off_east;
    *W(FDONE) = 1;
#else
    // ---- the control (NCRISC, NOC1)
    const uint32_t g = get_arg_val<uint32_t>(0), c = get_arg_val<uint32_t>(1), NCY = get_arg_val<uint32_t>(2);
    const uint32_t nexp_g = get_arg_val<uint32_t>(3);
    const auto cs = TensorAccessor(cargs, get_arg_val<uint32_t>(4), NEXP * 4);
    const auto ts = TensorAccessor(targs, get_arg_val<uint32_t>(5), TB);
    const auto dls = TensorAccessor(dargs, get_arg_val<uint32_t>(6), RMAX * 32 * 4);
    const bool split = get_arg_val<uint32_t>(10) != 0;
    uint32_t sx[GCN];
    for (uint32_t j = 0; j < GCN; ++j) {
        sx[j] = get_arg_val<uint32_t>(11 + j);
    }
    noc_async_read_page(0, cs, CNT + 64);
    noc_async_read_page(g, ts, TAB);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* counts = W(CNT);
    volatile tt_l1_ptr uint32_t* order = W(TAB) + 16;
    uint32_t n = 0, k = 0, t_arr = 0, t_dst = 0, t_wr = 0, t_cr = 0, t_issue = 0, t_sent = 0;
    for (uint32_t d_n = 0; d_n < nexp_g; ++d_n) {
        const uint32_t e = order[d_n] & 0xFFFF, M = counts[16 + e], rows = (M + 31) / 32;
        for (uint32_t d_r = 0; d_r < rows; ++d_r, ++n) {
            if (n % NCY != c) {
                continue;
            }
            const uint32_t s = k % NSL, need = GCN * (k / NSL + 1);
            const uint32_t a = wall();
            noc_async_read(dls.get_noc_addr(e) + d_r * 128, DBUF, 128);
            do {
                invalidate_l1_cache();
            } while (*W(YW + s * 16) < need);
            const uint32_t b = wall();
            noc_async_read_barrier();
            const uint32_t b2 = wall();
            const uint32_t m = M - d_r * 32, nr = m < 32 ? m : 32;
            if (!SINK) {
                if (split) {
                    *W(NRW) = nr | s << 8;
                    *W(GO) = k + 1;
                    do {
                        invalidate_l1_cache();
                    } while (*W(SENT) < k + 1);
                } else {
                    write_rows(s, nr, t_issue, t_sent);
                }
            }
            const uint32_t d = wall();
            for (uint32_t j = 0; j < GCN; ++j) {
                noc_inline_dw_write(get_noc_addr(sx[j] & 0xFF, sx[j] >> 8, CRED + c * 16), k + 1);
            }
            t_cr += wall() - d;
            t_arr += b - a;
            t_dst += b2 - b;
            t_wr += d - b2;
            ++k;
        }
    }
    const uint32_t f = wall();
    if (split) {
        *W(GO) = END;
        do {
            invalidate_l1_cache();
        } while (*W(FDONE) == 0);
    } else if (!SINK) {
        fence();
    }
    st[8] = wall() - f;
    noc_semaphore_inc(get_noc_addr(sx[0] & 0xFF, sx[0] >> 8, CRED + 3 * 16), 1);
    noc_async_atomic_barrier();
    noc_async_write_barrier();
    st[0] = t_start;
    st[2] = t_arr;
    st[3] = t_dst;
    st[4] = t_wr;
    st[5] = t_cr;
    st[6] = k;
    st[9] = t_issue;
    st[10] = t_sent;
    st[7] = wall();
    st[1] = st[7] - t_start;
#endif
}
