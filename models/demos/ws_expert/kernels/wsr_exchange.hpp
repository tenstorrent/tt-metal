// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert head: the routing outputs from every core's records through an exchange in L1 (NCRISC, NOC1: L1 exchanges
// and posted DRAM writes; no DRAM reads). Every core is a source (its records t | k << 16 | e << 20) and, for core ids
// below NOWN, the owner of experts [EPO me, EPO (me + 1)).
//   B. The records sorted by owner (stable), each owner's run padded to a multiple of 4 (16 B); padded counts to every
//      owner (word me of its CNT_IN, + 1 marks it set); an owner turns them into offsets (sources in core order) and
//      returns them (word o of each source's OFF_IN); sources write their runs into the owners' record buffers, then
//      + 1 on each owner's arrival semaphore once acknowledged. Each CNT_IN / OFF_IN word is written once a launch and
//      cleared once read, so the buffers are zero again for the next launch.
//   C. Owner: its records by expert, stable, into per-expert lists of ceil(M / 32) 32 entries: tokens t (padded with
//      T, x's zero row) and result rows t S + k; each list to its expert's page of the token lists / destination lists
//      (posted); its counts into core 0's CNT_ALL, then + 1 on core 0's done semaphore; core 0 writes the counts page
//      (posted). An owner whose records exceed CAP writes count 0xFFFFFFFF for its experts.
//   Fence: an acknowledged 4-B write into every DRAM bank (the fence tensor's pages) after the posted writes, same NoC
//   and VC: every write landed when the kernel exits.
// The including kernel's CT args: 0 K, 1 NE, 2 S, 3 EPO, 4 NOWN, 5 NC, 6 CAP, 7 RS (list page entries), 8 wst, 9 rec,
// 10 srt, 11 cnt_in, 12 off_in, 13 recbuf, 14 la, 15 lb, 16 cnt_all, 17 aux (3 x 256 words: per-owner counts, run
// starts, fill cursors - off the small stack), then TensorAccessorArgs(w), (idx), (dest), (counts), (fence).
// Semaphores 0 (arrivals) and 1 (done).
#pragma once

#include "api/dataflow/dataflow_api.h"

namespace wsr {

constexpr uint32_t PAD = 0xFFFFFFFFu;

template <typename T>
FORCE_INLINE volatile tt_l1_ptr T* p(uint32_t a) {
    return reinterpret_cast<volatile tt_l1_ptr T*>(a);
}

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

// a tile's element (r, c), 4-byte elements, faces of 16 x 16
FORCE_INLINE uint32_t at(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }

constexpr uint32_t K = get_compile_time_arg_val(0);
constexpr uint32_t NE = get_compile_time_arg_val(1);
constexpr uint32_t S = get_compile_time_arg_val(2);
constexpr uint32_t EPO = get_compile_time_arg_val(3);
constexpr uint32_t NOWN = get_compile_time_arg_val(4);
constexpr uint32_t NC = get_compile_time_arg_val(5);
constexpr uint32_t CAP = get_compile_time_arg_val(6);
constexpr uint32_t RS = get_compile_time_arg_val(7);
constexpr uint32_t wst = get_compile_time_arg_val(8);
constexpr uint32_t rec = get_compile_time_arg_val(9);
constexpr uint32_t srt = get_compile_time_arg_val(10);
constexpr uint32_t cnt_in = get_compile_time_arg_val(11);
constexpr uint32_t off_in = get_compile_time_arg_val(12);
constexpr uint32_t recbuf = get_compile_time_arg_val(13);
constexpr uint32_t la = get_compile_time_arg_val(14);
constexpr uint32_t lb = get_compile_time_arg_val(15);
constexpr uint32_t cnt_all = get_compile_time_arg_val(16);
constexpr uint32_t aux = get_compile_time_arg_val(17);
constexpr auto wargs = TensorAccessorArgs<18>();
constexpr auto iargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
constexpr auto dargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
constexpr auto cargs = TensorAccessorArgs<dargs.next_compile_time_args_offset()>();
constexpr auto fargs = TensorAccessorArgs<cargs.next_compile_time_args_offset()>();
constexpr uint32_t ct_end = fargs.next_compile_time_args_offset();  // (the including kernel's own CT args from here)
static_assert(NOWN < 256);

// B, C and the fence for this core's nrec records at rec. noc_at(c, addr): core c's NoC address of L1 addr. idx_addr ..
// fence_addr: the outputs' DRAM addresses. Stamps 3 (B done), 4 (C done), 63 (end).
template <typename NocAt>
void exchange(
    uint32_t me,
    uint32_t nrec,
    uint32_t T,
    NocAt noc_at,
    uint32_t idx_addr,
    uint32_t dest_addr,
    uint32_t counts_addr,
    uint32_t fence_addr,
    volatile tt_l1_ptr uint32_t* stamps) {
    const auto is = TensorAccessor(iargs, idx_addr, RS * 4);
    const auto ds = TensorAccessor(dargs, dest_addr, RS * 4);
    const auto cs = TensorAccessor(cargs, counts_addr, NE * 4);
    const auto fs = TensorAccessor(fargs, fence_addr, 64);

    // ---- B: sources
    volatile tt_l1_ptr uint32_t* cnt = p<uint32_t>(aux);
    volatile tt_l1_ptr uint32_t* start = p<uint32_t>(aux + 256 * 4);
    volatile tt_l1_ptr uint32_t* fill = p<uint32_t>(aux + 512 * 4);
    for (uint32_t o = 0; o < NOWN; ++o) {
        cnt[o] = 0;
    }
    for (uint32_t i = 0; i < nrec; ++i) {
        ++cnt[(p<uint32_t>(rec)[i] >> 20) / EPO];
    }
    start[0] = 0;
    for (uint32_t o = 0; o < NOWN; ++o) {
        start[o + 1] = start[o] + ((cnt[o] + 3) & ~3u);
    }
    for (uint32_t o = 0; o < NOWN; ++o) {
        fill[o] = start[o];
    }
    for (uint32_t i = 0; i < nrec; ++i) {
        const uint32_t v = p<uint32_t>(rec)[i];
        p<uint32_t>(srt)[fill[(v >> 20) / EPO]++] = v;
    }
    for (uint32_t o = 0; o < NOWN; ++o) {
        for (uint32_t i = fill[o]; i < start[o + 1]; ++i) {
            p<uint32_t>(srt)[i] = PAD;
        }
    }
    for (uint32_t o = 0; o < NOWN; ++o) {
        noc_inline_dw_write(noc_at(o, cnt_in + me * 4), start[o + 1] - start[o] + 1);
    }
    // ---- B: owners
    uint32_t total = 0;
    if (me < NOWN) {
        for (uint32_t c = 0; c < NC; ++c) {
            do {
                invalidate_l1_cache();
            } while (p<uint32_t>(cnt_in)[c] == 0);
        }
        for (uint32_t c = 0; c < NC; ++c) {
            noc_inline_dw_write(noc_at(c, off_in + me * 4), total + 1);
            total += p<uint32_t>(cnt_in)[c] - 1;
            p<uint32_t>(cnt_in)[c] = 0;
        }
    }
    // ---- B: sources, the runs
    for (uint32_t o = 0; o < NOWN; ++o) {
        do {
            invalidate_l1_cache();
        } while (p<uint32_t>(off_in)[o] == 0);
        const uint32_t off = p<uint32_t>(off_in)[o] - 1, n = start[o + 1] - start[o];
        p<uint32_t>(off_in)[o] = 0;
        if (n && off + n <= CAP) {
            noc_async_write(srt + start[o] * 4, noc_at(o, recbuf + off * 4), n * 4);
        }
    }
    noc_async_write_barrier();
    const uint32_t arr = get_semaphore(0), done = get_semaphore(1);
    for (uint32_t o = 0; o < NOWN; ++o) {
        noc_semaphore_inc(noc_at(o, arr), 1);
    }
    noc_async_atomic_barrier();
    stamps[3] = wall();

    // ---- C: owners
    if (me < NOWN) {
        noc_semaphore_wait(p<uint32_t>(arr), NC);
        const uint32_t e0 = me * EPO, ne = NE - e0 < EPO ? NE - e0 : EPO;
        const bool over = total > CAP;
        uint32_t m[EPO], base[EPO + 1], ef[EPO];
        for (uint32_t j = 0; j < EPO; ++j) {
            m[j] = 0;
        }
        if (!over) {
            for (uint32_t i = 0; i < total; ++i) {
                const uint32_t v = p<uint32_t>(recbuf)[i];
                if (v != PAD) {
                    ++m[(v >> 20) - e0];
                }
            }
        }
        base[0] = 0;
        for (uint32_t j = 0; j < ne; ++j) {
            base[j + 1] = base[j] + ((m[j] + 31) & ~31u);
            ef[j] = base[j];
        }
        if (!over) {
            for (uint32_t i = 0; i < total; ++i) {
                const uint32_t v = p<uint32_t>(recbuf)[i];
                if (v != PAD) {
                    const uint32_t j = (v >> 20) - e0, t = v & 0xFFFF;
                    p<uint32_t>(la)[ef[j]] = t;
                    p<uint32_t>(lb)[ef[j]] = t * S + ((v >> 16) & 0xF);
                    ++ef[j];
                }
            }
            for (uint32_t j = 0; j < ne; ++j) {
                for (uint32_t i = ef[j]; i < base[j + 1]; ++i) {
                    p<uint32_t>(la)[i] = T;
                    p<uint32_t>(lb)[i] = T * S;
                }
                const uint32_t bytes = (base[j + 1] - base[j]) * 4;
                if (bytes) {
                    noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                        la + base[j] * 4, is.get_noc_addr(e0 + j), bytes);
                    noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                        lb + base[j] * 4, ds.get_noc_addr(e0 + j), bytes);
                }
            }
        }
        for (uint32_t j = 0; j < ne; ++j) {
            noc_inline_dw_write(noc_at(0, cnt_all + (e0 + j) * 4), over ? PAD : m[j]);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(noc_at(0, done), 1);
        noc_async_atomic_barrier();
    }
    stamps[4] = wall();
    if (me == 0) {
        noc_semaphore_wait(p<uint32_t>(done), NOWN);
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(cnt_all, cs.get_noc_addr(0), NE * 4);
    }
    // ---- fence
    noc_async_posted_writes_flushed();
    for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
        noc_async_write(wst, fs.get_noc_addr(i), 4);
    }
    noc_async_write_barrier();
    stamps[63] = wall();
}

}  // namespace wsr
