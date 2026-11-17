// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert fold (the shared expert folded into the end pass, ws_fold.py), end core reader (BRISC, NOC0: DRAM reads,
// the shared rows' local copies, the slots' credits to the writers). End core e of cluster c owns the cluster's row
// tiles i = e, e + NE, .. (global r = c + NCL i); the k-th's half h (16 tokens) is its half m = 2 k + h, in ring slot
// h, written by the cluster's GC cores (wsh_reader.cpp, WSH_FOLD: each its columns of the 16 rows, then its flag word
// m + 1). Per token t of half m (once all GC flags of the slot read m + 1): its K routed rows (pages t K + j of the row-major routed rows) into CB
// rows, row j at tile slots 4 j .. 4 j + 3 (a row of H elements read contiguously into 4 1024-element slots), the
// shared row from the slot into row K, and its K gate weights (fp32, page t of w) as bf16 at element 0 of K scalar
// tiles of CB w, then 1.0 for the shared row (we_compute.cpp sums them). Up to NBUF tokens in flight on transaction
// ids. Once a half's last token landed, m + 1 into word e of every writer's FCRED (inline writes).
// Zeroed once: a row fills 3.75 slots, so the last 512 B of each row's 4th slot stay zero, as do the scalar tiles'
// other elements.
// A launch starts with this core's flags zeroed, then + 1 on the master's (end core 0 of cluster 0) semaphore 2; the
// master waits for every writer and end core, then sets semaphore 3 on each; all wait for it (the fold's start
// barrier: no flag or credit lands before its word is cleared).
// RT: 0 rows addr, 1 w addr, 2 c, 3 e, 4 stamps (3: cycles waiting on the reads, 4: for CB space, 7: for the
//     slots' flags), 5 + j: writer j's NoC x | y << 8 (GC), then the master's, then (the master) the barrier's
//     member count and every member's NoC x | y << 8.
// CT: 0 K, 1 row bytes, 2 w scratch L1 (64 B aligned, NBUF pages of 64 B), 3 NBUF, 4 CB w bytes, 5 NE, 6 NCL, 7 row
//     tiles, 8 T, 9 the end ring, 10 its flags, 11 GC, 12 FCRED, 13 CB rows, 14 CB w, then
//     TensorAccessorArgs(rows), (w).
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

FORCE_INLINE uint16_t bf16_rne(uint32_t f) { return static_cast<uint16_t>((f + 0x7FFF + ((f >> 16) & 1)) >> 16); }

FORCE_INLINE void zero_l1(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t w_scr = get_compile_time_arg_val(2);
    constexpr uint32_t NBUF = get_compile_time_arg_val(3);
    constexpr uint32_t w_cb_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t NE = get_compile_time_arg_val(5);
    constexpr uint32_t NCL = get_compile_time_arg_val(6);
    constexpr uint32_t R = get_compile_time_arg_val(7);
    constexpr uint32_t T = get_compile_time_arg_val(8);
    constexpr uint32_t ering = get_compile_time_arg_val(9);
    constexpr uint32_t eflags = get_compile_time_arg_val(10);
    constexpr uint32_t GC = get_compile_time_arg_val(11);
    constexpr uint32_t fcred = get_compile_time_arg_val(12);
    constexpr uint32_t rows_cb = get_compile_time_arg_val(13);
    constexpr uint32_t w_cb = get_compile_time_arg_val(14);
    constexpr auto rargs = TensorAccessorArgs<15>();
    constexpr auto wargs = TensorAccessorArgs<rargs.next_compile_time_args_offset()>();
    constexpr uint32_t S = K + 1, TPR = 4, tile_bytes = 2048, tok_pages = S * TPR, tok_bytes = tok_pages * tile_bytes;
    static_assert(row_bytes <= TPR * tile_bytes && NBUF >= 1 && NBUF < 16);

    const uint32_t c = get_arg_val<uint32_t>(2), e = get_arg_val<uint32_t>(3);
    const auto rs = TensorAccessor(rargs, get_arg_val<uint32_t>(0), row_bytes);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(1), 64);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    auto word = [](uint32_t base, uint32_t i) -> volatile tt_l1_ptr uint32_t& {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + i * 16);
    };
    // my tokens: q -> row tile k = q / 32 (global c + NCL (e + NE k)), token 32 r + q % 32
    auto rtile = [&](uint32_t k) { return c + NCL * (e + NE * k); };
    uint32_t ntok = 0;
    for (uint32_t k = 0; rtile(k) < R; ++k) {
        const uint32_t t0 = rtile(k) * 32;
        ntok += T - t0 < 32 ? (T > t0 ? T - t0 : 0) : 32;
    }
    auto token = [&](uint32_t q) { return rtile(q / 32) * 32 + q % 32; };
    auto last_of_half = [&](uint32_t q) { return q % 16 == 15 || q + 1 == ntok; };

    const uint32_t rows_base = get_write_ptr(rows_cb);
    for (uint32_t j = 0; j < NBUF * S; ++j) {
        zero_l1(rows_base + j * TPR * tile_bytes + row_bytes, TPR * tile_bytes - row_bytes);
    }
    zero_l1(get_write_ptr(w_cb), w_cb_bytes);
    const uint64_t me = get_noc_addr(my_x[noc_index], my_y[noc_index], 0);
    {  // the start barrier
        for (uint32_t j = 0; j < 2 * GC; ++j) {
            word(eflags, j) = 0;
        }
        const uint32_t mxy = get_arg_val<uint32_t>(5 + GC);
        const uint32_t cnt = get_semaphore(2), go = get_semaphore(3);
        noc_semaphore_inc(get_noc_addr(mxy & 0xFF, mxy >> 8, cnt), 1);
        noc_async_atomic_barrier();
        if (c == 0 && e == 0) {
            const uint32_t n = get_arg_val<uint32_t>(6 + GC);
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt), n);
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t v = get_arg_val<uint32_t>(7 + GC + j);
                noc_inline_dw_write(get_noc_addr(v & 0xFF, v >> 8, go), 1);
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go), 1);
    }
    uint32_t t_bar = 0, t_res = 0, t_flag = 0;
    uint32_t issued = 0, landed = 0;
    while (landed < ntok) {
        while (issued < ntok && issued - landed < NBUF) {
            const uint32_t i = issued % 32, h = i / 16, m = 2 * (issued / 32) + h, t = token(issued);
            if (i % 16 == 0) {  // (the slot's rows all in: every writer's flag)
                const uint32_t a = wall();
                for (uint32_t j = 0; j < GC; ++j) {
                    do {
                        invalidate_l1_cache();
                    } while (word(eflags, h * GC + j) < m + 1);
                }
                t_flag += wall() - a;
            }
            const uint32_t a = wall();
            cb_reserve_back(rows_cb, (issued - landed + 1) * tok_pages);
            t_res += wall() - a;
            const uint32_t slot = issued % NBUF, l1 = rows_base + slot * tok_bytes;
            noc_async_read_set_trid(1 + slot);
#if defined(WF_SLICES)
            // (diagnostic: each routed row read as WF_SLICES equal pieces - the reads a column-slice-major layout of the
            // routed rows would take - same bytes, same destination)
            static_assert(row_bytes % (WF_SLICES * 64) == 0, "WF_SLICES pieces of 64 B multiples");
            for (uint32_t j = 0; j < K; ++j) {
                const uint64_t src = rs.get_noc_addr(t * K + j);
                for (uint32_t q = 0; q < WF_SLICES; ++q) {
                    noc_async_read(
                        src + q * (row_bytes / WF_SLICES),
                        l1 + j * TPR * tile_bytes + q * (row_bytes / WF_SLICES),
                        row_bytes / WF_SLICES);
                }
            }
#else
            for (uint32_t j = 0; j < K; ++j) {
                noc_async_read_page(t * K + j, rs, l1 + j * TPR * tile_bytes);
            }
#endif
            noc_async_read(me | (ering + (h * 16 + i % 16) * row_bytes), l1 + K * TPR * tile_bytes, row_bytes);
            noc_async_read_page(t, ws, w_scr + slot * 64);
            noc_async_read_set_trid(0);
            ++issued;
        }
        const uint32_t slot = landed % NBUF;
        const uint32_t a = wall();
        noc_async_read_barrier_with_trid(1 + slot);
        t_bar += wall() - a;
        cb_reserve_back(w_cb, S);
        const uint32_t wl = get_write_ptr(w_cb);
        volatile tt_l1_ptr uint32_t* wv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_scr + slot * 64);
        for (uint32_t j = 0; j < K; ++j) {
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wl + j * tile_bytes) = bf16_rne(wv[j]);
        }
#ifdef WF_NO_SHARED
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wl + K * tile_bytes) = 0;  // (a check: the routed sum alone)
#else
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wl + K * tile_bytes) = 0x3F80;
#endif
        cb_push_back(w_cb, S);
        cb_push_back(rows_cb, tok_pages);
        if (last_of_half(landed)) {  // (the slot's shared rows copied out: free for half m + 2)
            const uint32_t m = 2 * (landed / 32) + (landed % 32) / 16;
            for (uint32_t j = 0; j < GC; ++j) {
                const uint32_t v = get_arg_val<uint32_t>(5 + j);
                noc_inline_dw_write(get_noc_addr(v & 0xFF, v >> 8, fcred + e * 16), m + 1);
            }
        }
        ++landed;
    }
    noc_async_write_barrier();
    stamps[3] = t_bar;
    stamps[4] = t_res;
    stamps[7] = t_flag;
}
