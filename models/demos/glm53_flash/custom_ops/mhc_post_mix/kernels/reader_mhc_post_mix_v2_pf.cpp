// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// The "prefetch" reader of the v2 post-mix program (op.py `MhcPostMix(prefetch="mcast")`, GLM53_MHC_POST_PREFETCH):
// the CB protocol of reader_mhc_post_mix_v2.cpp (the compute and writer kernels are shared, untouched), a different
// DRAM access pattern. One token-tile only (R == 1: T = 32, the decode shape; the host refuses anything else).
//
// Why (P1 of the decode roofline, chip (0,0) profile 2026-11-03: 29 us per wrap for ~4.3 MB of useful bytes): the
// plain reader has EVERY core read the token-tile's coefficient tiles from DRAM before it issues its stream reads --
// lane mode n + 1 tiles = 20 KB per core, 120 cores = 2.4 MB of identical reads from the 5 pages that hold them, which
// interleave onto DRAM banks 0-3 (comb pages 0..3, post page 0: bank 0 alone serves ~1 MB of them); EXPAND mode 8 KB
// per core = 960 KB, all from bank 0. With that hot spot bank 0 moves ~1.5 MB (coefficients + its 1/8 share of x, y
// and out) at ~64 GB/s = ~24 us while banks 4-7 idle: the op is bound by one DRAM bank, not by the aggregate rate.
// Here:
//   1. the stream tiles x[r, i, c] and y[r, c] of the first DEPTH units are issued first (pages c, Ct + c, ... :
//      spread over all 8 banks, ~2.2 MB in ~4.5 us at the DRAM peak), so they drain while the coefficients travel;
//   2. the coefficient tiles are read from DRAM once, by the leader core (logical (0,0)), and NoC-multicast into the
//      same CB address on every other core (one 20 KB / 8 KB L1 -> L1 multicast per grid rectangle). The handshake is
//      the matmul in0 one: a receiver resets its VALID flag and bumps the leader's READY counter, the leader waits for
//      every receiver, writes the payload (linked) then the VALID flag (unlinked, on the same path: ordered), the
//      receiver waits for VALID and pushes. The CB address is the same on every core: one CB layout per program and
//      nothing pushed into that CB before the reserve (the CB holds exactly one coefficient set);
//   3. the units (positions, then the extras of the "extras" split) are fetched DEPTH at a time: one reserve, every
//      read in flight, one barrier, one push -- the X / Y double buffer holds exactly DEPTH = 2 units.
// Bytes per core: 4 x tiles + y (18 KB) per unit from DRAM, 20 KB (8 KB) of coefficients by multicast; 9 reads in
// flight per core for the first batch; predicted ~15 us per wrap at the decode shape (README "Prefetch reader").
// Every output tile is still produced by the same core from the same operands in the same order: bit-identical.
//
// Runtime args: [x, y, post, comb, start, count] (+ the extras tail [n_extra, p0, j0, ...] when MHC_MAX_EXTRA > 0)
// + [is_leader, leader_noc_x, leader_noc_y, n_rects, (x_start, y_start, x_end, y_end, num_dests) x PF_MAX_RECTS]
// (rectangles in ascending NoC coordinates; `num_dests` excludes the leader when it sits inside: the non-loopback
// multicast skips the sender). Compile-time args: [CB_X, CB_Y, CB_RAW, CB_COEF, n, Ct, expand, hmode, SEM_READY,
// SEM_VALID] + TensorAccessorArgs(x, y, post, comb).
#ifndef MHC_MAX_EXTRA
#define MHC_MAX_EXTRA 0
#endif

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t y_addr = get_arg_val<uint32_t>(1);
    const uint32_t post_addr = get_arg_val<uint32_t>(2);
    const uint32_t comb_addr = get_arg_val<uint32_t>(3);
    const uint32_t start_pos = get_arg_val<uint32_t>(4);
    const uint32_t num_pos = get_arg_val<uint32_t>(5);
    constexpr uint32_t MAX_EXTRA = MHC_MAX_EXTRA;
    constexpr uint32_t EXTRA_BASE = 6;  // [n_extra, p0, j0, p1, j1, ...]
    constexpr uint32_t PF_BASE = EXTRA_BASE + (MAX_EXTRA > 0 ? 1 + 2 * MAX_EXTRA : 0);
    constexpr uint32_t PF_MAX_RECTS = 2;   // the column-major prefix of the grid: the full columns + a partial column
    constexpr uint32_t PF_RECT_WORDS = 5;  // x_start, y_start, x_end, y_end, num_dests
    constexpr uint32_t DEPTH = 2;          // units in flight: the X / Y CBs hold two
    const bool leader = get_arg_val<uint32_t>(PF_BASE) != 0;
    const uint32_t leader_x = get_arg_val<uint32_t>(PF_BASE + 1);
    const uint32_t leader_y = get_arg_val<uint32_t>(PF_BASE + 2);
    const uint32_t n_rects_arg = get_arg_val<uint32_t>(PF_BASE + 3);
    const uint32_t n_rects = n_rects_arg < PF_MAX_RECTS ? n_rects_arg : PF_MAX_RECTS;

    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_raw = get_compile_time_arg_val(2);
    constexpr uint32_t cb_coef = get_compile_time_arg_val(3);
    constexpr uint32_t n = get_compile_time_arg_val(4);
    constexpr uint32_t Ct = get_compile_time_arg_val(5);
    constexpr bool expand = get_compile_time_arg_val(6) != 0;
    constexpr bool hmode = get_compile_time_arg_val(7) != 0;  // pre-mix: the coefficient tile is `post` (= pre), no y
    constexpr uint32_t sem_ready = get_compile_time_arg_val(8);
    constexpr uint32_t sem_valid = get_compile_time_arg_val(9);
    constexpr auto x_args = TensorAccessorArgs<10>();
    constexpr auto y_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto post_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    constexpr auto comb_args = TensorAccessorArgs<post_args.next_compile_time_args_offset()>();
    // the coefficient CB of the token-tile: the raw tiles (CB_RAW, expanded by the compute kernel) or the lane tiles
    constexpr uint32_t cb_c = expand ? cb_raw : cb_coef;
    constexpr uint32_t coef_tiles = hmode ? 1 : (expand ? 2 : n + 1);

    const uint32_t x_page = get_local_cb_interface(cb_x).fifo_page_size;
    const uint32_t y_page = get_local_cb_interface(cb_y).fifo_page_size;
    const uint32_t coef_page = get_local_cb_interface(cb_c).fifo_page_size;

    const auto s_x = TensorAccessor(x_args, x_addr);
    const auto s_y = TensorAccessor(y_args, y_addr);
    const auto s_post = TensorAccessor(post_args, post_addr);
    const auto s_comb = TensorAccessor(comb_args, comb_addr);

    Noc noc;
    CircularBuffer cbx(cb_x), cby(cb_y), cbc(cb_c);

    uint32_t n_extra = 0;
    if constexpr (MAX_EXTRA > 0) {
        n_extra = get_arg_val<uint32_t>(EXTRA_BASE);
        n_extra = n_extra < MAX_EXTRA ? n_extra : MAX_EXTRA;
    }
    const uint32_t n_units = num_pos + n_extra;

    // the position of unit u: the core's positions, then its extras (the compute / writer order)
    auto unit_pos = [&](uint32_t u) -> uint32_t {
        if (u < num_pos) {
            return start_pos + u;
        }
        return get_arg_val<uint32_t>(EXTRA_BASE + 1 + 2 * (u - num_pos));
    };
    // reserve slots for units [u0, u0 + cnt) and issue their reads (slot k <- unit u0 + k)
    auto issue_units = [&](uint32_t u0, uint32_t cnt) {
        cbx.reserve_back(cnt * n);
        if constexpr (!hmode) {
            cby.reserve_back(cnt);
        }
        for (uint32_t k = 0; k < cnt; ++k) {
            const uint32_t p = unit_pos(u0 + k);
            const uint32_t r = p / Ct;
            const uint32_t c = p - r * Ct;
            for (uint32_t i = 0; i < n; ++i) {
                noc.async_read(
                    s_x, cbx, x_page, {.page_id = (r * n + i) * Ct + c}, {.offset_bytes = (k * n + i) * x_page});
            }
            if constexpr (!hmode) {
                noc.async_read(s_y, cby, y_page, {.page_id = r * Ct + c}, {.offset_bytes = k * y_page});
            }
        }
    };
    auto push_units = [&](uint32_t cnt) {
        cbx.push_back(cnt * n);
        if constexpr (!hmode) {
            cby.push_back(cnt);
        }
    };

    // ---- 1. the first batch of stream reads goes out before anything else
    const uint32_t first = n_units < DEPTH ? n_units : DEPTH;
    if (first) {
        issue_units(0, first);
    }

    // ---- 2. the coefficients of token-tile 0: DRAM once (leader) -> multicast into every core's reserved slots
    cbc.reserve_back(coef_tiles);
    const uint32_t coef_l1 = get_write_ptr(cb_c);  // the same address on every core (nothing pushed into cb_c yet)
    const uint32_t coef_bytes = coef_tiles * coef_page;
    const uint32_t valid_addr = get_semaphore(sem_valid);
    volatile tt_l1_ptr uint32_t* p_valid = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(valid_addr);
    if (leader) {
        if constexpr (hmode) {
            noc.async_read(s_post, cbc, coef_page, {.page_id = 0}, {.offset_bytes = 0});
        } else if constexpr (expand) {
            noc.async_read(s_comb, cbc, coef_page, {.page_id = 0}, {.offset_bytes = 0});
            noc.async_read(s_post, cbc, coef_page, {.page_id = 0}, {.offset_bytes = coef_page});
        } else {
            for (uint32_t j = 0; j < n; ++j) {
                noc.async_read(s_comb, cbc, coef_page, {.page_id = j}, {.offset_bytes = j * coef_page});
            }
            noc.async_read(s_post, cbc, coef_page, {.page_id = 0}, {.offset_bytes = n * coef_page});
        }
        noc.async_read_barrier();  // the coefficient tiles landed (the first batch too: one counter)
        if (n_rects) {
            uint32_t receivers = 0;
            for (uint32_t q = 0; q < n_rects; ++q) {
                receivers += get_arg_val<uint32_t>(PF_BASE + 4 + q * PF_RECT_WORDS + 4);
            }
            volatile tt_l1_ptr uint32_t* p_ready =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ready));
            noc_semaphore_wait_min(p_ready, receivers);  // every receiver reset its flag and reserved its slots
            noc_semaphore_set(p_ready, 0);
            noc_semaphore_set(p_valid, 1);  // the flag multicast reads this core's cell
            for (uint32_t q = 0; q < n_rects; ++q) {
                const uint32_t base = PF_BASE + 4 + q * PF_RECT_WORDS;
                uint32_t xs = get_arg_val<uint32_t>(base), ys = get_arg_val<uint32_t>(base + 1);
                uint32_t xe = get_arg_val<uint32_t>(base + 2), ye = get_arg_val<uint32_t>(base + 3);
                const uint32_t num_dests = get_arg_val<uint32_t>(base + 4);
                if constexpr (noc_index == 1) {
                    // a NOC_1 multicast rectangle runs high -> low (the matmul program factories swap the corners)
                    const uint32_t tx = xs, ty = ys;
                    xs = xe;
                    ys = ye;
                    xe = tx;
                    ye = ty;
                }
                // the payload linked, the flag unlinked right behind it: same NoC, same path, in order
                noc_async_write_multicast(
                    coef_l1, get_noc_multicast_addr(xs, ys, xe, ye, coef_l1), coef_bytes, num_dests, /*linked=*/true);
                noc_semaphore_set_multicast(
                    valid_addr, get_noc_multicast_addr(xs, ys, xe, ye, valid_addr), num_dests, /*linked=*/false);
            }
        }
        cbc.push_back(coef_tiles);
    } else {
        noc_semaphore_set(p_valid, 0);
        noc_semaphore_inc(get_noc_addr(leader_x, leader_y, get_semaphore(sem_ready)), 1);
        noc_semaphore_wait(p_valid, 1);
        cbc.push_back(coef_tiles);
    }

    // ---- 3. the units, DEPTH at a time (the next batch's reserve waits for the compute kernel to free the slots)
    uint32_t done = 0;
    uint32_t cnt = first;
    while (cnt) {
        noc.async_read_barrier();
        push_units(cnt);
        done += cnt;
        const uint32_t left = n_units - done;
        cnt = left < DEPTH ? left : DEPTH;
        if (cnt) {
            issue_units(done, cnt);
        }
    }
    // every NoC transaction (the multicasts, the semaphore atomics and the reads) acknowledged before the kernel
    // returns: a late ack after the next program's NOC init leaves its counters unequal forever
    noc_async_full_barrier();
}
