// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// GDN decode step v2, writer (fused_step/gdn_step_v2.py). Core (h, j):
// 1. conv shift register of the channel tiles this core owns (full tiles, exactly the composite's four ttnn.copy
//    results st0 <- st1 <- st2 <- st3 <- x): read them all, wait until every core of the key group has finished
//    reading the shared history (sem_rd == G), write them all; no per-tile round trips
// 2. the head's RMS partial sum(o_j^2): inline 4-byte writes into slot j of every head peer's gather tile (and its
//    own), then +1 on each peer's sem_x; waits for the Vt - 1 peers and hands the gather tile to the compute core
// 3. the updated state tiles S[h][t][j] (in place) and the gated output tile h * Vt + j
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

constexpr uint32_t cb_wscr = 4, cb_out = 5, cb_state = 27, cb_ps = 30, cb_gather = 31;
constexpr uint32_t BF = 2048, F32 = 4096;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t sem_rd = get_compile_time_arg_val(2);
    constexpr uint32_t sem_x = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr auto out_args = TensorAccessorArgs<5>();
    constexpr auto rec_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto st0_args = TensorAccessorArgs<rec_args.next_compile_time_args_offset()>();
    constexpr auto st1_args = TensorAccessorArgs<st0_args.next_compile_time_args_offset()>();
    constexpr auto st2_args = TensorAccessorArgs<st1_args.next_compile_time_args_offset()>();
    constexpr auto st3_args = TensorAccessorArgs<st2_args.next_compile_time_args_offset()>();
    constexpr auto proj_args = TensorAccessorArgs<st3_args.next_compile_time_args_offset()>();

    uint32_t i = 0;
    const uint32_t out_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rec_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st0_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st1_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st2_addr = get_arg_val<uint32_t>(i++);
    const uint32_t st3_addr = get_arg_val<uint32_t>(i++);
    const uint32_t proj_addr = get_arg_val<uint32_t>(i++);
    const uint32_t h = get_arg_val<uint32_t>(i++);
    const uint32_t j = get_arg_val<uint32_t>(i++);
    const uint32_t n_own = get_arg_val<uint32_t>(i++);
    const uint32_t own_arg = i;
    const uint32_t peer_arg = own_arg + n_own;  // Vt - 1 (x, y) pairs

    const auto out = TensorAccessor(out_args, out_addr, BF);
    const auto rec = TensorAccessor(rec_args, rec_addr, F32);
    const auto st0 = TensorAccessor(st0_args, st0_addr, BF);
    const auto st1 = TensorAccessor(st1_args, st1_addr, BF);
    const auto st2 = TensorAccessor(st2_args, st2_addr, BF);
    const auto st3 = TensorAccessor(st3_args, st3_addr, BF);
    const auto proj = TensorAccessor(proj_args, proj_addr, BF);

    // ---- 1. conv shift register of the owned channels ----
    {
        DeviceZoneScopedN("W2-shift");
        cb_reserve_back(cb_wscr, 4 * n_own);
        const uint32_t scr = get_write_ptr(cb_wscr);
        for (uint32_t o = 0; o < n_own; o++) {
            const uint32_t c = get_arg_val<uint32_t>(own_arg + o);
            const uint32_t d = scr + 4 * o * BF;
            noc_async_read_page(c, st1, d);
            noc_async_read_page(c, st2, d + BF);
            noc_async_read_page(c, st3, d + 2 * BF);
            noc_async_read_page(c, proj, d + 3 * BF);
        }
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* rd = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rd));
        noc_semaphore_wait_min(rd, G);
        noc_semaphore_set(rd, 0);
        for (uint32_t o = 0; o < n_own; o++) {
            const uint32_t c = get_arg_val<uint32_t>(own_arg + o);
            const uint32_t d = scr + 4 * o * BF;
            noc_async_write_page(c, st0, d);
            noc_async_write_page(c, st1, d + BF);
            noc_async_write_page(c, st2, d + 2 * BF);
            noc_async_write_page(c, st3, d + 3 * BF);
        }
    }

    // ---- 2. RMS partial-sum exchange among the Vt cores of head h ----
    {
        DeviceZoneScopedN("W2-xchg");
        cb_reserve_back(cb_gather, 1);
        const uint32_t gather = get_write_ptr(cb_gather);
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

    // ---- 3. updated state (in place) ----
    cb_wait_front(cb_state, Kt);
    {
        uint32_t src = get_read_ptr(cb_state);
        for (uint32_t t = 0; t < Kt; t++, src += F32) {
            noc_async_write_page((h * Kt + t) * Vt + j, rec, src);
        }
    }
    {
        DeviceZoneScopedN("W2-xwait");
        volatile tt_l1_ptr uint32_t* xs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_x));
        noc_semaphore_wait_min(xs, Vt - 1);
        noc_semaphore_set(xs, 0);
        cb_push_back(cb_gather, 1);
    }

    // ---- gated output ----
    cb_wait_front(cb_out, 1);
    noc_async_write_page(h * Vt + j, out, get_read_ptr(cb_out));
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
    cb_pop_front(cb_state, Kt);
}
