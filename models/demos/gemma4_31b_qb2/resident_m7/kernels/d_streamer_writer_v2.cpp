// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter streamer writer v2 (BRISC; M6b fix 2, M6B_TIMING_1.md §5; d_streamer_writer.cpp is the
// verified v1): each coefficient column tile goes only to the slices whose channels it covers (slice k: column tiles
// g0_k .. g1_k of each section, g = channel tile / 16), at the slice's local index 3 s + (g - g0_k), and only those
// slices' sem_coef move (d_slice_dm_v2.cpp counts its senders). Timeline marks 5 / 6: the attention / MLP coefficients
// sent. Fix 3: the o / down partials' tile bytes are compile-time arg 58 (2048 bf16, or 1088 BF8 partials: the
// compute packs cb_pout in its CB's format); the staging, slot and offset arithmetic uses them.
//
// v1: see d_streamer_common.hpp: the activation plumbing of one
// streamer core, streamer_writer.cpp (the target's, M4 stub path) plus the conv-coefficient sends. Per layer:
//   take h: send this core's coefficient tiles into every slice's coefficient buffer (at their column offset), bump
//     every slice's sem_coef (every core bumps, also with no tiles)
//   take xin: o = q (attention stub): this core's q columns into every streamer's o_in, wait for all, release o_in;
//     send the o_proj partial into its slice's slot of this chip and the hub's staging (round 2 l)
//   take h2: the MLP coefficients, as above
//   take m_in: silu(g) * u into every streamer's act, wait for all, release act; the down partial (round 2 l + 1)
//   then: the logits' argmax records [0, kMw) to the hub, and (dump) the raw logits to DRAM.
// The activation buffer cb_h is rewritten by the slices only after a dependency chain through this core's use of the
// previous one (the coefficients of h precede xin; the all-reduces precede h2 and the next h), so it is not doubled.
// Mailboxes as streamer_writer.cpp: streamer 0 publishes cb_h's address to the slices, the hub its staging address, and
// slice 0 the slots and the coefficient buffer addresses to the streamers.
//
// Runtime args: 0 ng, 1 act_off, 2 nd, 3 pout_off, 4 hub_x, 5 hub_y, 6 nc (coefficient tiles), 7 hub records address,
// 8 hub sem_gather id, 9 hub sem_rec id, 10 nq, 11 q0, 12 c0 (first coefficient tile), 13 sem_coefaddr id (the slices'
// coefficient buffer address mailbox), 14 logits dump (DRAM, 0 = off), 15 nv, 16 vocab index of the first logit,
// 17 this core's index, 18 timeline (0 = off), 19 slice_x, 20 slice_y, 21 sem_stage id, 22 / 23 slice slot semaphores
// (parity 0 / 1), 24 first column within its slice, 25 slice slot bytes (per chip), 26 sem_slotaddr id, 27 slices'
// sem_haddr id, then num_streamers x (x, y) of the streamers, then K x (x, y) of the slices.

#include "api/dataflow/dataflow_api.h"
#include "d_streamer_common.hpp"

using namespace dres;

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t act_off = get_arg_val<uint32_t>(1);
    const uint32_t nd = get_arg_val<uint32_t>(2);
    const uint32_t pout_off = get_arg_val<uint32_t>(3);
    const uint32_t hub_x = get_arg_val<uint32_t>(4);
    const uint32_t hub_y = get_arg_val<uint32_t>(5);
    const uint32_t nc = get_arg_val<uint32_t>(6);
    const uint32_t hub_rec = get_arg_val<uint32_t>(7);
    const uint32_t hub_gather = get_semaphore(get_arg_val<uint32_t>(8));
    const uint32_t hub_rec_sem = get_semaphore(get_arg_val<uint32_t>(9));
    const uint32_t nq = get_arg_val<uint32_t>(10);
    const uint32_t q0 = get_arg_val<uint32_t>(11);
    const uint32_t c0 = get_arg_val<uint32_t>(12);
    volatile tt_l1_ptr uint32_t* coefaddr_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(13)));
    const uint32_t logits_dump = get_arg_val<uint32_t>(14);
    const uint32_t nv = get_arg_val<uint32_t>(15);
    const uint32_t first_index = get_arg_val<uint32_t>(16);
    const uint32_t core = get_arg_val<uint32_t>(17);
    const uint32_t ts_addr = get_arg_val<uint32_t>(18);
    const uint32_t slice_x = get_arg_val<uint32_t>(19);
    const uint32_t slice_y = get_arg_val<uint32_t>(20);
    volatile tt_l1_ptr uint32_t* stage_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(21)));
    const uint32_t slice_sem[2] = {get_arg_val<uint32_t>(22), get_arg_val<uint32_t>(23)};
    const uint32_t slice_off = get_arg_val<uint32_t>(24);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(25);
    volatile tt_l1_ptr uint32_t* slots_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(26)));
    const uint32_t haddr_box = get_semaphore(get_arg_val<uint32_t>(27));
    constexpr uint32_t peers_base = 28;
    constexpr uint32_t slices_base = peers_base + 2 * num_streamers;
    constexpr uint32_t num_chips = 4;

    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    const uint32_t act_addr = cb_base(cb_act);  // the same on every streamer
    if (core == 0) {
        for (uint32_t k = 0; k < K; k++) {
            const uint32_t sx = get_arg_val<uint32_t>(slices_base + 2 * k);
            const uint32_t sy = get_arg_val<uint32_t>(slices_base + 2 * k + 1);
            noc_inline_dw_write(get_noc_addr(sx, sy, haddr_box), cb_base(cb_h));
        }
    }

    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    constexpr uint32_t kTsW = 8;
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[layers * 8 + l * kTsW + i] = *clk;  // after the reader's [0, 8 layers)
        }
    };
    volatile tt_l1_ptr uint32_t* slots_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_slots));
    volatile tt_l1_ptr uint32_t* o_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_o));
    volatile tt_l1_ptr uint32_t* act_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_act));
    const uint32_t act_sem_addr = get_semaphore(sem_act);
    const uint64_t gather_noc = get_noc_addr(hub_x, hub_y, hub_gather);
    constexpr uint32_t kPartBytes = get_compile_time_arg_val(58);  // fix 3: a partial tile's bytes
    const uint32_t staging_bytes = Ht * kPartBytes;
    uint32_t hub_staging = 0, slice_slots = 0, slice_coef = 0;

    // the slices' activation m (1-based) is complete once each of the K slices has bumped sem_slots m times
    auto take_h = [&](uint32_t m) {
        noc_semaphore_wait_min(slots_sem, K * m);
        cb_reserve_back(cb_h, Ht);
        cb_push_back(cb_h, Ht);
    };
    auto send_partial = [&](uint32_t r) {
        cb_wait_front(cb_pout, nd_max);
        while (hub_staging == 0 || slice_slots == 0) {
            invalidate_l1_cache();
            hub_staging = *stage_box;
            slice_slots = *slots_box;
        }
        const uint32_t par = r & 1;
        const uint32_t src = get_read_ptr(cb_pout);
        const uint32_t to_slice = slice_slots + (par * num_chips + chip) * slot_bytes + slice_off * kPartBytes;
        noc_async_write(src, get_noc_addr(slice_x, slice_y, to_slice), nd * kPartBytes);
        noc_async_write(
            src, get_noc_addr(hub_x, hub_y, hub_staging + par * staging_bytes + pout_off * kPartBytes), nd * kPartBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(slice_x, slice_y, slice_sem[par]), 1);
        noc_semaphore_inc(gather_noc, 1);
        cb_pop_front(cb_pout, nd_max);
    };
    // this core's coefficient tiles into the slices that use them (fix 2), then bump those slices' sem_coef
    constexpr uint32_t T = Ht / K;
    constexpr uint32_t kSection = 12, kLocal = 3;
    auto send_coef = [&]() {
        cb_wait_front(cb_coef, kBlkC);
        while (slice_coef == 0) {
            invalidate_l1_cache();
            slice_coef = *coefaddr_box;
        }
        uint32_t used = 0;  // bit k: slice k got a tile
        for (uint32_t i = 0; i < nc; i++) {
            const uint32_t c = c0 + i, s = c / kSection, g = c % kSection;
            for (uint32_t k = 0; k < K; k++) {
                const uint32_t g0 = (k * T) / 16, g1 = (k * T + T - 1) / 16;
                if (g < g0 || g > g1) {
                    continue;
                }
                const uint32_t sx = get_arg_val<uint32_t>(slices_base + 2 * k);
                const uint32_t sy = get_arg_val<uint32_t>(slices_base + 2 * k + 1);
                noc_async_write(get_read_ptr(cb_coef) + i * kTileBytes,
                                get_noc_addr(sx, sy, slice_coef + (kLocal * s + g - g0) * kTileBytes), kTileBytes);
                used |= 1u << k;
            }
        }
        noc_async_write_barrier();
        for (uint32_t k = 0; k < K; k++) {
            if (used & (1u << k)) {
                const uint32_t sx = get_arg_val<uint32_t>(slices_base + 2 * k);
                const uint32_t sy = get_arg_val<uint32_t>(slices_base + 2 * k + 1);
                noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(sem_coef)), 1);
            }
        }
        cb_pop_front(cb_coef, kBlkC);
    };
    // n tiles at src into every streamer's buffer at dst (+ off tiles), then bump their semaphore (rotated order)
    auto all_gather = [&](uint32_t src, uint32_t n, uint32_t dst, uint32_t off, uint32_t sem_addr) {
        if (n > 0) {
            for (uint32_t i = 1; i <= num_streamers; i++) {
                const uint32_t s = (core + i) % num_streamers;
                const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
                noc_async_write(src, get_noc_addr(px, py, dst + off * kTileBytes), n * kTileBytes);
            }
            noc_async_write_barrier();
        }
        for (uint32_t i = 1; i <= num_streamers; i++) {
            const uint32_t s = (core + i) % num_streamers;
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, sem_addr), 1);
        }
    };

    for (uint32_t l = 0; l < layers; l++) {
        // attention half
        take_h(4 * l + 1);
        mark(l, 0);
        send_coef();
        mark(l, 5);
        take_h(4 * l + 2);
        mark(l, 1);
        cb_wait_front(cb_q, kBlk);
        // o = q: this core's q columns [q0, q0 + nq) within the Ot q tiles into every streamer's o_in
        const uint32_t nsend = q0 >= kOt ? 0 : (q0 + nq <= kOt ? nq : kOt - q0);
        all_gather(get_read_ptr(cb_q), nsend, act_addr, q0, get_semaphore(sem_o));
        cb_pop_front(cb_q, kBlk);
        noc_semaphore_wait_min(o_sem, num_streamers * (l + 1));
        cb_reserve_back(cb_o_in, kAct);
        cb_push_back(cb_o_in, kAct);
        send_partial(2 * l);
        mark(l, 2);
        // mlp half
        take_h(4 * l + 3);
        send_coef();
        mark(l, 6);
        take_h(4 * l + 4);
        mark(l, 3);
        cb_wait_front(cb_aslice, kBlk);
        all_gather(get_read_ptr(cb_aslice), ng, act_addr, act_off, act_sem_addr);
        cb_pop_front(cb_aslice, kBlk);
        noc_semaphore_wait_min(act_sem, num_streamers * (l + 1));
        cb_reserve_back(cb_act, kAct);
        cb_push_back(cb_act, kAct);
        send_partial(2 * l + 1);
        mark(l, 4);
    }
    // LM head: rows [0, kMw) of this core's logits as records on the hub; the raw logits to the dump
    take_h(4 * layers + 1);
    cb_wait_front(cb_logits, nv_max);
    const uint32_t stage = get_read_ptr(cb_pout);  // idle once the last partial left
    argmax_rows(get_read_ptr(cb_logits), nv, first_index, 0, kMw, reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage));
    noc_async_write(stage, get_noc_addr(hub_x, hub_y, hub_rec + core * kRecStride), kMw * 8);
    noc_async_write_barrier();
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, hub_rec_sem), 1);
    if (logits_dump) {
        const InterleavedAddrGen<true> dump{.bank_base_address = logits_dump, .page_size = nv_max * kTileBytes};
        noc_async_write(get_read_ptr(cb_logits), dump.get_noc_addr(core), nv_max * kTileBytes);
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
}
