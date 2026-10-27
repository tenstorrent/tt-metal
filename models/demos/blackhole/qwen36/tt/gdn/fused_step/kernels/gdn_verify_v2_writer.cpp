// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN fused verify v2, writer (fused_step/gdn_verify_v2.py). Core (h, j):
// 0. E_new of the owned channel tiles (v2's ownership: q/k channel c of the key group -> member c % G, v_j -> (h, j)):
//    rows 0..2 = E_prev rows mi+1..mi+3, rows 3..3+T-1 = proj rows 0..T-1, other rows 0 (the composite's one-hot
//    matmul result). Built from the reader's L1 copies, written into _verify_win_buf in place once every reader of the
//    key group has loaded E_prev (sem_rd == G).
// per token t (v2 steps 2-3 + output):
// 1. the head's RMS partial sum(o_j^2) into slot j of gather tile t % 2 of every head peer (and its own), +1 on each
//    peer's sem_x; wait for sem_x >= (t + 1) (Vt - 1) (cumulative: a peer one token ahead never resets a count),
//    hand gather tile t % 2 to the compute core
// 2. token t's state -> ring block t * Nv + h (value-column block j)
// 3. row 0 of the gated output tile -> row t of the assembled output tile; at the end the tile -> out[h * Vt + j]
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_wscr = 4, cb_out = 5, cb_state = 27, cb_ps = 30, cb_gather = 31;
constexpr uint32_t cb_ep = 34, cb_pj = 35, cb_oasm = 36, cb_idx = 37;
constexpr uint32_t BF = 2048, F32 = 4096;

FORCE_INLINE uint32_t bf_row_w(uint32_t r, uint32_t f) { return ((r >= 16 ? 2 : 0) + f) * 128 + (r % 16) * 8; }

FORCE_INLINE void copy_row(uint32_t src, uint32_t rs, uint32_t dst, uint32_t rd) {
    volatile tt_l1_ptr uint32_t* ps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    volatile tt_l1_ptr uint32_t* pd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t f = 0; f < 2; f++) {
        const uint32_t so = bf_row_w(rs, f), dof = bf_row_w(rd, f);
        for (uint32_t e = 0; e < 8; e++) {
            pd[dof + e] = ps[so + e];
        }
    }
}

FORCE_INLINE void zero_row(uint32_t dst, uint32_t rd) {
    volatile tt_l1_ptr uint32_t* pd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t f = 0; f < 2; f++) {
        const uint32_t dof = bf_row_w(rd, f);
        for (uint32_t e = 0; e < 8; e++) {
            pd[dof + e] = 0;
        }
    }
}

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t sem_rd = get_compile_time_arg_val(2);
    constexpr uint32_t sem_x = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr uint32_t T = get_compile_time_arg_val(5);
    constexpr uint32_t Nv = get_compile_time_arg_val(6);
    constexpr auto out_args = TensorAccessorArgs<7>();
    constexpr auto ring_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto win_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr uint32_t NS = 2 * Kt + 1;
    constexpr uint32_t KC1 = 3;

    uint32_t i = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t ring_addr = get_arg_val<uint32_t>(i++);
    const uint32_t win_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);
    const uint32_t n_own = get_arg_val<uint32_t>(i++);
    const uint32_t own_arg = i;                     // n_own (slot, channel) pairs
    const uint32_t peer_arg = own_arg + 2 * n_own;  // Vt - 1 (x, y) pairs

    const auto out = TensorAccessor(out_args, out_addr, BF);
    const auto ring = TensorAccessor(ring_args, ring_addr, F32);
    const auto win = TensorAccessor(win_args, win_addr, BF);

    const uint32_t gather0 = get_write_ptr(cb_gather);  // both gather tiles: gather0, gather0 + F32

    // ---- 0. E_new of the owned channels ----
    {
        DeviceZoneScopedN("WV-window");
        cb_wait_front(cb_idx, 1);
        const uint32_t mi = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_idx))[0] / Nv;
        cb_wait_front(cb_ep, NS);
        cb_wait_front(cb_pj, NS + 1);
        const uint32_t ep = get_read_ptr(cb_ep), pj = get_read_ptr(cb_pj);
        cb_reserve_back(cb_wscr, n_own);
        const uint32_t scr = get_write_ptr(cb_wscr);
        for (uint32_t o = 0; o < n_own; o++) {
            const uint32_t s = get_arg_val<uint32_t>(own_arg + 2 * o);
            const uint32_t d = scr + o * BF;
            for (uint32_t r = 0; r < 32; r++) {
                if (r < KC1) {
                    copy_row(ep + s * BF, mi + 1 + r, d, r);
                } else if (r < KC1 + T) {
                    copy_row(pj + s * BF, r - KC1, d, r);
                } else {
                    zero_row(d, r);
                }
            }
        }
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rd));
        noc_semaphore_wait_min(rd, G);
        noc_semaphore_set(rd, 0);
        for (uint32_t o = 0; o < n_own; o++) {
            const uint32_t c = get_arg_val<uint32_t>(own_arg + 2 * o + 1);
            noc_async_write_page(c, win, scr + o * BF);
        }
    }

    // assembled output tile, zero
    cb_reserve_back(cb_oasm, 1);
    const uint32_t oasm = get_write_ptr(cb_oasm);
    for (uint32_t r = 0; r < 32; r++) {
        zero_row(oasm, r);
    }

    volatile tt_l1_ptr uint32_t* xs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x));
    for (uint32_t t = 0; t < T; t++) {
        // ---- 1. RMS partial-sum exchange among the Vt cores of head h (gather tile t % 2) ----
        {
            DeviceZoneScopedN("WV-xchg");
            cb_reserve_back(cb_gather, 1);
            const uint32_t gather = gather0 + (t & 1) * F32;
            cb_wait_front(cb_ps, 1);
            const uint32_t val = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_ps))[0];
            cb_pop_front(cb_ps, 1);
            for (uint32_t p = 0; p < Vt - 1; p++) {
                const uint32_t px = get_arg_val<uint32_t>(peer_arg + 2 * p);
                const uint32_t py = get_arg_val<uint32_t>(peer_arg + 2 * p + 1);
                noc_inline_dw_write(get_noc_addr(px, py, gather + 4 * j), val);
            }
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gather)[j] = val;
            noc_async_write_barrier();
            for (uint32_t p = 0; p < Vt - 1; p++) {
                const uint32_t px = get_arg_val<uint32_t>(peer_arg + 2 * p);
                const uint32_t py = get_arg_val<uint32_t>(peer_arg + 2 * p + 1);
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(sem_x)), 1);
            }
        }
        // ---- 2. token t's state -> ring block t * Nv + h ----
        cb_wait_front(cb_state, Kt);
        {
            uint32_t src = get_read_ptr(cb_state);
            const uint32_t blk = t * Nv + h;
            for (uint32_t kt = 0; kt < Kt; kt++, src += F32) {
                noc_async_write_page((blk * Kt + kt) * Vt + j, ring, src);
            }
        }
        {
            DeviceZoneScopedN("WV-xwait");
            noc_semaphore_wait_min(xs, (t + 1) * (Vt - 1));
            cb_push_back(cb_gather, 1);
        }
        // ---- 3. gated output row t ----
        cb_wait_front(cb_out, 1);
        copy_row(get_read_ptr(cb_out), 0, oasm, t);
        cb_pop_front(cb_out, 1);
        noc_async_write_barrier();
        cb_pop_front(cb_state, Kt);
    }
    noc_async_write_page(h * Vt + j, out, oasm);
    noc_async_write_barrier();
    noc_semaphore_set(xs, 0);
    cb_pop_front(cb_ep, NS);
    cb_pop_front(cb_pj, NS + 1);
    cb_pop_front(cb_idx, 1);
}
