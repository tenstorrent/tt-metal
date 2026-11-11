// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident streamer writer (BRISC): the activation plumbing of one streamer core. Per layer:
//   mixer half: take the slices' h; send this core's q|k|v columns into the rx buffer of the attention leader of each
//     column (sliding: the 4 group leaders, kv head g's q heads 2g, 2g + 1 and its k, v; full: leader 0) and bump
//     their A_RX; once the leaders' O rows are in o_in (sem_o: 4 per sliding layer, 1 per full layer), release o_in to
//     compute; send the o_proj partial (this core's hidden columns) into its slice's slot of this chip (local
//     reduction) and into the hub's staging (the peer chips).
//   mlp half: take h; send gelu_tanh(g) * u into every streamer's act and bump their sem_act; release act; send the
//     down partial the same way.
//   then: per logit row [0, kMw) the largest logit of this core (order key, vocab index) into its record on the hub.
// Every buffer is rewritten only after a dependency chain through this core's use of it, so none is double-buffered.
// The big buffers are CBs (every core of a role has the same layout): streamer 0 publishes the streamers' h address
// to the slices, the hub its staging address and slice 0 the slots address to the streamers (local semaphores used as
// mailboxes, as the original's published hub slots).
// Adapted from changh95/qwen3.8-27b-p300x2 code/qwen36_resident/kernels/streamer_writer.cpp (Apache-2.0).
//
// Runtime args: 0 ng, 1 act_off (first gate column of this core in It), 2 nd, 3 pout_off (first hidden column),
// 4 hub_x, 5 hub_y, 6 sem_rxaddr id (mailbox of the leaders' rx address), 7 hub records address, 8 hub sem_gather id, 9 hub sem_rec id, 10 nq sliding,
// 11 q0 sliding, 12 nq full, 13 q0 full (this core's projection tiles and its first tile in the row), 14 logits
// (serving: DRAM TILE [1, 1, 32, VOCAB / chips] bf16 per chip, raw head output rows 0 .. R - 1; 0 = none),
// 15 nv, 16 vocab index of this core's first logit, 17 this core's index, 18 timeline buffer (0 = off), 19 slice_x,
// 20 slice_y, 21 sem_stage id (hub staging address mailbox), 22 / 23 slice slot semaphores (parity 0 / 1), 24 this
// core's first column within its slice, 25 slice slot bytes (per chip), 26 sem_slotaddr id (slots address mailbox),
// 27 slices' sem_haddr id, 28 logits dump (DRAM, 0 = off), 29 sem_ctrl id (write the logits: set by the reader at
// launch), then num_streamers x (x, y) of the streamers, then
// K x (x, y) of the slices, then the 4 attention leaders (x | y << 16). Streamer 0 publishes o_in's address to the
// leaders (their A_OADDR mailbox). (The q and act gathers unicast to every streamer: 32 concurrent multicasts into the
// same rectangles measured 1.6x slower at M = 1 and 1.2x at M = 32, contending with the weight reads.)

#include "api/dataflow/dataflow_api.h"
#include "streamer_common.hpp"

using namespace gres;

namespace {
constexpr uint32_t num_chips = 4;
constexpr uint32_t kLeaders = 4;
constexpr uint32_t A_RX = 0, A_OADDR = 5;  // the attention cores' semaphore ids (attn_common.hpp)

// projection tile c of the layer -> (leader, index in its rx): sliding [q 16 | k 8 | v 8] of kv head g, full all
FORCE_INLINE void rx_slot(bool full, uint32_t c, uint32_t& g, uint32_t& idx) {
    if (full) {
        g = 0;
        idx = c;
    } else if (c < 64) {
        g = c / 16;
        idx = c % 16;
    } else if (c < 96) {
        g = (c - 64) / 8;
        idx = 16 + (c - 64) % 8;
    } else {
        g = (c - 96) / 8;
        idx = 24 + (c - 96) % 8;
    }
}
}  // namespace

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t act_off = get_arg_val<uint32_t>(1);
    const uint32_t nd = get_arg_val<uint32_t>(2);
    const uint32_t pout_off = get_arg_val<uint32_t>(3);
    const uint32_t hub_x = get_arg_val<uint32_t>(4);
    const uint32_t hub_y = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* rxaddr_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(6)));
    const uint32_t hub_rec = get_arg_val<uint32_t>(7);
    const uint32_t hub_gather = get_semaphore(get_arg_val<uint32_t>(8));
    const uint32_t hub_rec_sem = get_semaphore(get_arg_val<uint32_t>(9));
    const uint32_t nq_s = get_arg_val<uint32_t>(10);
    const uint32_t q0_s = get_arg_val<uint32_t>(11);
    const uint32_t nq_f = get_arg_val<uint32_t>(12);
    const uint32_t q0_f = get_arg_val<uint32_t>(13);
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
    const uint32_t logits_dump = get_arg_val<uint32_t>(28);
    const uint32_t logits_out = get_arg_val<uint32_t>(14);
    const uint32_t ctrl_box = get_semaphore(get_arg_val<uint32_t>(29));
    constexpr uint32_t peers_base = 30;
    constexpr uint32_t slices_base = peers_base + 2 * num_streamers;
    constexpr uint32_t leaders_base = slices_base + 2 * K;
    auto leader = [&](uint32_t g, uint32_t local) {
        const uint32_t xy = get_arg_val<uint32_t>(leaders_base + g);
        return get_noc_addr(xy & 0xffff, xy >> 16, local);
    };

    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    const uint32_t act_addr = cb_base(cb_act);  // the same on every streamer
    if (core == 0) {
        // the streamers' h address to every slice, o_in's to every attention leader
        for (uint32_t k = 0; k < K; k++) {
            const uint32_t sx = get_arg_val<uint32_t>(slices_base + 2 * k);
            const uint32_t sy = get_arg_val<uint32_t>(slices_base + 2 * k + 1);
            noc_inline_dw_write(get_noc_addr(sx, sy, haddr_box), cb_base(cb_h));
        }
        for (uint32_t g = 0; g < kLeaders; g++) {
            noc_inline_dw_write(leader(g, get_semaphore(A_OADDR)), act_addr);
        }
    }

    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * kTsWords + 8 + i] = *clk;
        }
    };
    volatile tt_l1_ptr uint32_t* slots_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_slots));
    volatile tt_l1_ptr uint32_t* o_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_o));
    volatile tt_l1_ptr uint32_t* act_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_act));
    const uint32_t act_sem_addr = get_semaphore(sem_act);
    const uint64_t gather_noc = get_noc_addr(hub_x, hub_y, hub_gather);
    const uint32_t staging_bytes = Ht * kTileBytes;
    uint32_t hub_staging = 0, slice_slots = 0;

    // h of round m (1-based) is complete once each of the K slices has bumped sem_slots m times
    auto take_h = [&](uint32_t m) {
        noc_semaphore_wait_min(slots_sem, K * m);
        cb_reserve_back(cb_h, Ht);
        cb_push_back(cb_h, Ht);
    };
    // the partial of all-reduce round r (0-based): this chip's slot [r & 1][chip] of the slice, and the hub's staging
    auto send_partial = [&](uint32_t r) {
        cb_wait_front(cb_pout, nd_max);
        while (hub_staging == 0 || slice_slots == 0) {
            invalidate_l1_cache();
            hub_staging = *stage_box;
            slice_slots = *slots_box;
        }
        const uint32_t par = r & 1;
        const uint32_t src = get_read_ptr(cb_pout);
        const uint32_t to_slice = slice_slots + (par * num_chips + chip) * slot_bytes + slice_off * kTileBytes;
        noc_async_write(src, get_noc_addr(slice_x, slice_y, to_slice), nd * kTileBytes);
        noc_async_write(
            src, get_noc_addr(hub_x, hub_y, hub_staging + par * staging_bytes + pout_off * kTileBytes), nd * kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(slice_x, slice_y, slice_sem[par]), 1);
        noc_semaphore_inc(gather_noc, 1);
        cb_pop_front(cb_pout, nd_max);
    };
    // this core's q|k|v tiles of the layer into the leaders' rx, then bump the leaders' A_RX
    uint32_t rx = 0;
    auto send_qkv = [&](bool full, uint32_t q0, uint32_t nq) {
        const uint32_t src = get_read_ptr(cb_q);
        while (rx == 0) {
            invalidate_l1_cache();
            rx = *rxaddr_box;
        }
        for (uint32_t i = 0; i < nq;) {
            uint32_t g, idx, run = 1;
            rx_slot(full, q0 + i, g, idx);
            for (; i + run < nq; run++) {
                uint32_t g2, idx2;
                rx_slot(full, q0 + i + run, g2, idx2);
                if (g2 != g || idx2 != idx + run) {
                    break;
                }
            }
            noc_async_write(src + i * kTileBytes, leader(g, rx + idx * kTileBytes), run * kTileBytes);
            i += run;
        }
        noc_async_write_barrier();
        for (uint32_t g = 0; g < (full ? 1 : kLeaders); g++) {
            noc_semaphore_inc(leader(g, get_semaphore(A_RX)), 1);
        }
    };
    // n tiles at src into every streamer's buffer at dst (+ off tiles), then bump their semaphore
    auto all_gather = [&](uint32_t src, uint32_t n, uint32_t dst, uint32_t off, uint32_t sem_addr) {
        if (n > 0) {
            for (uint32_t s = 0; s < num_streamers; s++) {
                const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
                noc_async_write(src, get_noc_addr(px, py, dst + off * kTileBytes), n * kTileBytes);
            }
            noc_async_write_barrier();
        }
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, sem_addr), 1);
        }
    };

    uint32_t o_expected = 0;
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t nq = full ? nq_f : nq_s;
        const uint32_t q0 = full ? q0_f : q0_s;
        // mixer half
        take_h(2 * l + 1);
        mark(l, 0);
        cb_wait_front(cb_q, kBlk);
        send_qkv(full, q0, nq);
        cb_pop_front(cb_q, kBlk);
        o_expected += full ? 1 : kLeaders;
        noc_semaphore_wait_min(o_sem, o_expected);
        mark(l, 1);
        cb_reserve_back(cb_o_in, kAct);
        cb_push_back(cb_o_in, kAct);
        send_partial(2 * l);
        mark(l, 2);
        // mlp half
        take_h(2 * l + 2);
        mark(l, 3);
        cb_wait_front(cb_aslice, kBlk);
        all_gather(get_read_ptr(cb_aslice), ng, act_addr, act_off, act_sem_addr);
        cb_pop_front(cb_aslice, kBlk);
        noc_semaphore_wait_min(act_sem, num_streamers * (l + 1));
        mark(l, 4);
        cb_reserve_back(cb_act, kAct);
        cb_push_back(cb_act, kAct);
        send_partial(2 * l + 1);
        mark(l, 5);
    }
    // LM head: rows [0, kMw) of this core's logits as records [core][R rows][key, index] on the hub
    take_h(2 * layers + 1);
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
    // serving: with the control word's write_logits, this core's raw logits rows into tile row 0 of the logits tensor
    // (the engine caps them in a separate op; rows R .. 31 are not written)
    if (logits_out && nv > 0) {
        invalidate_l1_cache();
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_box) != 0) {
            const InterleavedAddrGen<true> lg{.bank_base_address = logits_out, .page_size = 2048};
            const uint32_t t0 = (first_index % 65536) / 32;  // this core's first vocab tile in the chip's shard
            const uint32_t src = get_read_ptr(cb_logits);
            // per face its rows 0 .. R - 1; DRAM writes of at least 64 B (R = 1: face row 1 takes the next 32 B
            // of L1 too, a row nothing reads)
            constexpr uint32_t wb = 32 * R < 64 ? 64 : 32 * R;
            for (uint32_t j = 0; j < nv; j++) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(src + j * kTileBytes + f * 32 * R, lg.get_noc_addr(t0 + j, f * 512), wb);
                }
            }
            noc_async_write_barrier();
        }
    }
    // (cb_logits is not popped: the reader may still be scanning its rows)
    noc_async_atomic_barrier();
}
