// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident streamer writer (BRISC): the activation plumbing of one streamer core. Per layer:
//   mixer half: release the hub's partial slots; send this core's projection columns to every mixer
//     core of the layer (GDN: the head cores; attention: the leader and the tail core) and write a GDN
//     layer's newest conv input back to its history slot (the oldest one, see conv_step); release o once the
//     mixer cores have written it; send the out-proj columns to the hub.
//   mlp half: release the slots; exchange silu(g) * u slices with every streamer; send the down
//     columns to the hub.
// Every buffer is rewritten only after a dependency chain through this core's use of it, so none is
// double-buffered (e.g. a head's next o needs this core's next projection row, which needs the mlp
// slots, which need this core's down output, which comes after its out-proj used the current o).
//
// Runtime args: 0 ng, 1 act_off, 2 nd, 3 pout_off, 4 hub_x, 5 hub_y, 6 publish (this core tells the hub the
// streamers' slots address, which is the same on every streamer), 7 unused,
// 8 GDN projection tiles, 9 their tile offset in the GDN row, 10 attention projection tiles, 11 their
// offset in the attention row, 12 mixer row buffer address (same on every mixer core), 13 n_conv,
// 14 conv history address, 15 this core's history rows base, 16 GDN copies, 17 unused,
// then gdn_heads x (x, y) of the head cores, then 2 x (x, y) of the attention leader and tail core,
// then num_streamers x (x, y) of the streamers, then an optional timeline buffer (0 = off): per layer 8
// wall-clock words at the phase boundaries (slots, row ready, mixer done, attn partial sent, slots,
// act slice ready, act complete, mlp partial sent), after the reader's 8 words; then (lm_head) the
// argmax record address on the hub (0 = off), this core's valid logit columns and the vocab index of
// its first column, this core's index, then the GDN row runs: count, then per run (first tile of this
// core's projection block, tiles, first head core, head cores) — each head core gets only the tiles it
// reads (its key head's q | k, its value head's v | z, every head's a | b).
//
// The hub's slots address arrives in sem_addr before the first partial (the slots are a hub CB).
//
// With the lm_head the writer scans this core's logits for their maximum (the first one on ties) and
// writes (order key, vocab index) to record slot [core] on the hub, so the host reads num_streamers
// candidates per chip instead of the logits. With sample (batch 1) it writes this core's top sampling::kTopK
// instead, for the hub's on-device sampling (sampling.hpp).

#include "api/dataflow/dataflow_api.h"
#include "sampling.hpp"
#include "streamer_common.hpp"

using namespace resident;

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t act_off = get_arg_val<uint32_t>(1);
    const uint32_t nd = get_arg_val<uint32_t>(2);
    const uint32_t pout_off = get_arg_val<uint32_t>(3);
    const uint32_t hub_x = get_arg_val<uint32_t>(4);
    const uint32_t hub_y = get_arg_val<uint32_t>(5);
    const bool publish = get_arg_val<uint32_t>(6) != 0;
    const uint32_t nq_gdn = get_arg_val<uint32_t>(8);
    const uint32_t q_off_gdn = get_arg_val<uint32_t>(9);
    const uint32_t nq_attn = get_arg_val<uint32_t>(10);
    const uint32_t q_off_attn = get_arg_val<uint32_t>(11);
    const uint32_t mixer_rows = get_arg_val<uint32_t>(12);
    const uint32_t n_conv = get_arg_val<uint32_t>(13);
    const uint32_t hist_addr = get_arg_val<uint32_t>(14);
    const uint32_t first_row = get_arg_val<uint32_t>(15);
    const uint32_t gdn_copies = get_arg_val<uint32_t>(16);
    constexpr uint32_t heads_base = 18;
    constexpr uint32_t attn_base = heads_base + 2 * gdn_heads;
    constexpr uint32_t peers_base = attn_base + 4;
    const uint32_t ts_addr = get_arg_val<uint32_t>(peers_base + 2 * num_streamers);
    const uint32_t argmax_addr = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 1);
    const uint32_t valid_cols = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 2);
    const uint32_t first_index = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 3);
    const uint32_t core = get_arg_val<uint32_t>(peers_base + 2 * num_streamers + 4);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * kTsWords + 8 + i] = *clk;
        }
    };

    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    if (publish) {
        noc_inline_dw_write(get_noc_addr(hub_x, hub_y, get_semaphore(sem_addr)), cb_base(cb_x));
    }
    volatile tt_l1_ptr uint32_t* hub_addr_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_addr));
    uint32_t hub_slots = 0;
    const uint32_t act_addr = cb_base(cb_act);  // the same CB address on every streamer
    volatile tt_l1_ptr uint32_t* slots_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_slots));
    volatile tt_l1_ptr uint32_t* act_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_act));
    volatile tt_l1_ptr uint32_t* heads_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_heads));
    const uint32_t act_sem_addr = get_semaphore(sem_act);
    const uint32_t rows_sem_addr = get_semaphore(sem_rows);
    const uint64_t hub_gather_sem = get_noc_addr(hub_x, hub_y, get_semaphore(sem_gather));
    const InterleavedAddrGen<true> hist{.bank_base_address = hist_addr, .page_size = kHistSlots * kBlkBytes};
    // the ring step (it picks the history slot to overwrite) comes from the reader: this RISC reads nothing,
    // as the reader's NOC1 weight reads share the read counters of NOC1
    volatile tt_l1_ptr uint32_t* ring_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_ring));
    while (*ring_sem == 0) {
        invalidate_l1_cache();
    }
    const uint32_t ring = *ring_sem - 1;

    // x of round 0 (x0) is seeded and released by the reader; the hub multicasts x of the later rounds
    auto take_slots = [&](uint32_t round) {
        if (round > 0) {
            noc_semaphore_wait_min(slots_sem, round);
            cb_reserve_back(cb_x, Ht);
            cb_push_back(cb_x, Ht);
        }
    };
    auto send_partial = [&](uint32_t round) {
        cb_wait_front(cb_pout, nd_max);
        while (hub_slots == 0) {
            invalidate_l1_cache();
            hub_slots = *hub_addr_sem;
        }
        const uint32_t dst = hub_slots + (((round & 1) * num_chips + chip) * Ht + pout_off) * kTileBytes;
        noc_async_write(get_read_ptr(cb_pout), get_noc_addr(hub_x, hub_y, dst), nd * kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(hub_gather_sem, 1);
        cb_pop_front(cb_pout, nd_max);
    };
    auto send_row = [&](uint32_t first, uint32_t count, uint32_t nq, uint32_t q_off) {
        const uint32_t row = get_read_ptr(cb_qkvz_out);
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t hx = get_arg_val<uint32_t>(first + 2 * i);
            const uint32_t hy = get_arg_val<uint32_t>(first + 2 * i + 1);
            noc_async_write(row, get_noc_addr(hx, hy, mixer_rows + q_off * kTileBytes), nq * kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t i = 0; i < count; i++) {
            const uint32_t hx = get_arg_val<uint32_t>(first + 2 * i);
            const uint32_t hy = get_arg_val<uint32_t>(first + 2 * i + 1);
            noc_semaphore_inc(get_noc_addr(hx, hy, rows_sem_addr), 1);
        }
    };

    constexpr uint32_t runs_base = peers_base + 2 * num_streamers + 5;
    const uint32_t n_runs = get_arg_val<uint32_t>(runs_base);
    auto send_runs = [&](uint32_t q_off) {
        const uint32_t row = get_read_ptr(cb_qkvz_out);
        for (uint32_t r = 0; r < n_runs; r++) {
            const uint32_t start = get_arg_val<uint32_t>(runs_base + 1 + 4 * r);
            const uint32_t count = get_arg_val<uint32_t>(runs_base + 2 + 4 * r);
            const uint32_t first = get_arg_val<uint32_t>(runs_base + 3 + 4 * r);
            const uint32_t dests = get_arg_val<uint32_t>(runs_base + 4 + 4 * r);
            for (uint32_t c = first; c < first + dests; c++) {
                const uint32_t hx = get_arg_val<uint32_t>(heads_base + 2 * c);
                const uint32_t hy = get_arg_val<uint32_t>(heads_base + 2 * c + 1);
                noc_async_write(
                    row + start * kTileBytes,
                    get_noc_addr(hx, hy, mixer_rows + (q_off + start) * kTileBytes),
                    count * kTileBytes);
            }
        }
        noc_async_write_barrier();
        for (uint32_t r = 0; r < n_runs; r++) {
            const uint32_t first = get_arg_val<uint32_t>(runs_base + 3 + 4 * r);
            const uint32_t dests = get_arg_val<uint32_t>(runs_base + 4 + 4 * r);
            for (uint32_t c = first; c < first + dests; c++) {
                const uint32_t hx = get_arg_val<uint32_t>(heads_base + 2 * c);
                const uint32_t hy = get_arg_val<uint32_t>(heads_base + 2 * c + 1);
                noc_semaphore_inc(get_noc_addr(hx, hy, rows_sem_addr), 1);
            }
        }
    };

    if constexpr (fc_in) {
        send_partial(0);
    }
    uint32_t g = 0;
    uint32_t o_expected = 0;  // o writes so far: o_heads per GDN layer, the leader per attention layer
    for (uint32_t l = 0; l < layers; l++) {
        const bool attn = is_attn(l);
        // mixer half
        take_slots(2 * l + kRound0);
        mark(l, 0);
        cb_wait_front(cb_qkvz_out, kBlk);
        mark(l, 1);
        if (attn) {
            send_row(attn_base, 2, nq_attn, q_off_attn);
            o_expected += 1;
        } else {
            send_runs(q_off_gdn);
            o_expected += o_heads;
            constexpr uint32_t outs = verify ? 3 : 1;  // verify: T(k + 2), T(k + 3), T(k + 4)
            cb_wait_front(cb_hist_out, outs * kBlkViews);
            if (n_conv > 0) {
                // the 32x32 views holding the q|k|v tiles
                const uint32_t step = conv_step(ring, g, gdn_copies);
                for (uint32_t j = 0; j < outs; j++) {
                    noc_async_write(
                        get_read_ptr(cb_hist_out) + j * kBlkViews * 2048,
                        hist.get_noc_addr(
                            first_row + g % gdn_copies, ((step + (verify ? 2 + j : 0)) % kHistSlots) * kBlkBytes),
                        views_of(n_conv) * 2048);
                }
                noc_async_write_barrier();
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local)) = g + 1;
            cb_pop_front(cb_hist_out, outs * kBlkViews);
            g++;
        }
        cb_pop_front(cb_qkvz_out, kBlk);
        noc_semaphore_wait_min(heads_sem, o_expected);
        mark(l, 2);
        cb_reserve_back(cb_o_in, Ot);
        cb_push_back(cb_o_in, Ot);
        send_partial(2 * l + kRound0);
        mark(l, 3);

        // mlp half
        take_slots(2 * l + 1 + kRound0);
        mark(l, 4);
        cb_wait_front(cb_aslice, kBlk);
        mark(l, 5);
        const uint32_t slice = get_read_ptr(cb_aslice);
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_async_write(slice, get_noc_addr(px, py, act_addr + act_off * kTileBytes), ng * kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, act_sem_addr), 1);
        }
        cb_pop_front(cb_aslice, kBlk);
        noc_semaphore_wait_min(act_sem, num_streamers * (l + 1));
        mark(l, 6);
        cb_reserve_back(cb_act, It);
        cb_push_back(cb_act, It);
        send_partial(2 * l + 1 + kRound0);
        mark(l, 7);
    }
    if constexpr (lm_head) {
        take_slots(2 * layers + kRound0);
        if (argmax_addr) {
            // bf16 bits -> unsigned key in the order of the values; per user, record slot core * batch + u.
            // Column c of user u: tile c / 32, face (c % 32) / 16, face row u
            cb_wait_front(cb_logits, nv_max);
            const uint32_t base = get_read_ptr(cb_logits);
            auto key = [](uint32_t b) { return sampling::key_of(b); };
            volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_pout));
            if constexpr (sample) {
                static_assert(nd_max * kTileBytes >= sampling::kListBytes, "the list is staged in cb_pout");
                sampling::clear(rec);
                for (uint32_t i = 0; i < valid_cols; i += 2) {
                    const uint32_t c = i % 32;
                    const uint32_t w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        base + (i / 32) * kTileBytes + (c / 16) * 32)[(c % 16) / 2];
                    sampling::insert(rec, key(w & 0xffff), first_index + i);
                    if (i + 1 < valid_cols) {
                        sampling::insert(rec, key(w >> 16), first_index + i + 1);
                    }
                }
            }
            for (uint32_t u = 0; !sample && u < batch; u++) {
                uint32_t best = 0, best_i = 0;
                for (uint32_t i = 0; i < valid_cols; i += 2) {
                    const uint32_t c = i % 32;
                    const uint32_t w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                        base + (i / 32) * kTileBytes + ((c / 16) * batch + u) * 32)[(c % 16) / 2];
                    const uint32_t k0 = key(w & 0xffff);
                    if (k0 > best) {
                        best = k0;
                        best_i = i;
                    }
                    if (i + 1 < valid_cols) {
                        const uint32_t k1 = key(w >> 16);
                        if (k1 > best) {
                            best = k1;
                            best_i = i + 1;
                        }
                    }
                }
                rec[4 * u] = best;
                rec[4 * u + 1] = first_index + best_i;
            }
            constexpr uint32_t rec_bytes = sample ? sampling::kListBytes : batch * 16;
            noc_async_write(
                get_write_ptr(cb_pout), get_noc_addr(hub_x, hub_y, argmax_addr + core * rec_bytes), rec_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(hub_gather_sem, 1);  // the record is on the hub (a feeding hub waits for it)
        }
    }
    noc_async_atomic_barrier();
}
