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
// Rows mode (active_rows.hpp): a GDN layer writes its whole shifted history (h1, h2, y) back to slots 0..2;
// with the preserve word, only the active rows' face rows of its q|k|v tiles.
//
// With the lm_head the writer scans this core's logits for their maximum (the first one on ties) and
// writes (order key, vocab index) to record slot [core] on the hub, so the host reads num_streamers
// candidates per chip instead of the logits. With sample it writes this core's top sampling::kTopK instead, for
// the hub's on-device sampling (sampling.hpp): batch 1, or (rows mode) one list per row, the active rows' filled,
// laid out [row][streamer] on the hub.

#include "api/dataflow/dataflow_api.h"
#include "sampling.hpp"
#include "streamer_common.hpp"
#include "active_rows.hpp"

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
    uint32_t mixer_rows = get_arg_val<uint32_t>(12);  // (row groups: from the leader, read after the publishes below)
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
        if constexpr (kOinInAct) {
            // row groups: the o / activation memory (the same address on every streamer) to the mixer cores (a
            // compact loop: the streamer core's kernels sit close to the kernel config buffer's size)
            const uint32_t act_base = cb_base(cb_act);
#pragma GCC unroll 1
            for (uint32_t c = 0; c < gdn_heads; c++) {
                noc_inline_dw_write(
                    get_noc_addr(
                        get_arg_val<uint32_t>(heads_base + 2 * c),
                        get_arg_val<uint32_t>(heads_base + 2 * c + 1),
                        get_semaphore(sem_raddr)),
                    act_base);
            }
            noc_inline_dw_write(
                get_noc_addr(
                    get_arg_val<uint32_t>(attn_base), get_arg_val<uint32_t>(attn_base + 1), get_semaphore(sem_raddr)),
                act_base);
        }
    }
    if constexpr (groups > 1) {
        // row groups: the mixers' row buffer address, from the attention leader (active_rows.hpp kCbRows) — only after
        // this core's own publishes: the leader waits for streamer 0's before it publishes the rows
        volatile tt_l1_ptr uint32_t* raddr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_raddr));
        while (*raddr == 0) {
            invalidate_l1_cache();
        }
        mixer_rows = *raddr;
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
    // (rows mode: the active rows and the preserve word, see the reader)
    const uint32_t ring = rows_mode ? 0 : *ring_sem - 1;
    // (row groups: the active word has a bit per global row g * batch + u, more than the semaphore's 16; it is read
    // from the reader's copy of the token state head in cb_dout, which compute first writes after this core's first
    // o_in release below)
    invalidate_l1_cache();
    const uint32_t active_mask = groups == 1
                                     ? (*ring_sem & 0xffff)
                                     : reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb_dout))[kTokActive];
    // (row groups: group gr's conv history rows follow the num_streamers x gdn_copies rows of the groups before it)
    const uint32_t hist_group_rows = num_streamers * gdn_copies;
    // row groups: only the groups with an active row run (active_rows.hpp): their sub-rounds, rows, histories,
    // activation slices and records; an inactive group's CB pages are popped unsent
    const ActiveGroups ag(rows_mode ? active_mask : 0, batch);
    const bool preserve = rows_mode && ((*ring_sem >> 16) & 1);

    // x of round 0 (x0) is seeded and released by the reader; the hub multicasts x of the later rounds
    // (row groups: the hub runs `groups` sub-rounds per round and counts the slots semaphore per sub-round; x of
    // every group is in before the round's compute starts, group g at tile offset g * Ht)
    auto take_slots = [&](uint32_t round) {
        if (round > 0) {
            noc_semaphore_wait_min(slots_sem, round * ag.n);
            cb_reserve_back(cb_x, groups * Ht);
            cb_push_back(cb_x, groups * Ht);
        }
    };
    // group g's partial goes to the hub's slot of sub-round s = round * groups + g (parity s & 1). Two parities keep
    // a slot free for up to 2 groups: a group's slot is summed before the round's last multicast, which the next
    // round's partials wait for. More groups need a slot-free signal from the hub (static_assert below).
    // More groups (G > 2) reuse a slot inside one round: group k + 2's partial waits until the hub has multicast
    // sub-round sr - 2's x (it summed that slot first; the hub sets the slots semaphore to s + 1 after sub-round s).
    // Against the peer hubs two slots suffice for any G: a hub's send of sub-round t follows its wait for the peers'
    // t - 1, which they send after finishing t - 2.
    static_assert(groups == 1 || o_tiles == Ot, "the mixers' o block per group is the out projection's K");
    auto send_partial = [&](uint32_t round) {
        for (uint32_t g = 0, k = 0; g < groups; g++) {
            cb_wait_front(cb_pout, nd_max);
            if (!ag.has(g)) {
                cb_pop_front(cb_pout, nd_max);
                continue;
            }
            while (hub_slots == 0) {
                invalidate_l1_cache();
                hub_slots = *hub_addr_sem;
            }
            const uint32_t sr = round * ag.n + k++;  // the hub's sub-round: the k-th active group of the round
            if constexpr (groups > 2) {
                if (sr >= 2) {
                    noc_semaphore_wait_min(slots_sem, sr - 1);
                }
            }
            const uint32_t dst = hub_slots + (((sr & 1) * num_chips + chip) * Ht + pout_off) * kTileBytes;
            noc_async_write(get_read_ptr(cb_pout), get_noc_addr(hub_x, hub_y, dst), nd * kTileBytes);
            noc_async_write_barrier();
            // (row groups: parity-0 sub-rounds count in the gather word's low half, parity 1 in its high half)
            noc_semaphore_inc(hub_gather_sem, groups == 1 ? 1 : 1u << (16 * (sr & 1)));
            cb_pop_front(cb_pout, nd_max);
        }
    };
    // (row groups: group gr's projection block at gr * kBlk tiles of cb_qkvz_out goes to block gr of the mixers' row
    // buffer, row_tiles tiles per group; one rows semaphore increment per layer once every group is written)
    // (row groups: the attention row goes to the leader only — the tail core reads its k / v rows there — but both
    // get the rows semaphore increment)
    auto send_row = [&](uint32_t first, uint32_t count, uint32_t nq, uint32_t q_off) {
        const uint32_t writes = groups > 1 ? 1 : count;
        for (uint32_t gr = 0; gr < groups; gr++) {
            if (!ag.has(gr)) {
                continue;
            }
            const uint32_t row = get_read_ptr(cb_qkvz_out) + gr * kBlk * kTileBytes;
            for (uint32_t i = 0; i < writes; i++) {
                const uint32_t hx = get_arg_val<uint32_t>(first + 2 * i);
                const uint32_t hy = get_arg_val<uint32_t>(first + 2 * i + 1);
                noc_async_write(
                    row, get_noc_addr(hx, hy, mixer_rows + (gr * row_tiles + q_off) * kTileBytes), nq * kTileBytes);
            }
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
        for (uint32_t gr = 0; gr < groups; gr++) {
            if (!ag.has(gr)) {
                continue;
            }
            const uint32_t row = get_read_ptr(cb_qkvz_out) + gr * kBlk * kTileBytes;
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
                        get_noc_addr(hx, hy, mixer_rows + (gr * row_tiles + q_off + start) * kTileBytes),
                        count * kTileBytes);
                }
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
        cb_wait_front(cb_qkvz_out, groups * kBlk);
        mark(l, 1);
        if (attn) {
            send_row(attn_base, 2, nq_attn, q_off_attn);
            o_expected += 1;
        } else {
            send_runs(q_off_gdn);
            o_expected += o_heads;
            // verify: T(k + 2), T(k + 3), T(k + 4); rows mode: (h1, h2, y)
            constexpr uint32_t outs = verify_pair || rows_mode ? 3 : 1;
            // (row groups: group gr's outs slots at gr * outs * kBlkViews, its history rows hist_group_rows further)
            cb_wait_front(cb_hist_out, groups * outs * kBlkViews);
            if (verify_rows && n_conv > 0) {
                // input row r -> ring slot ring + 3 + r: rows 0 .. n1 - 1 to block b from row o, the rest to block
                // b + 1 from row 0, per tile and face (see streamer_common.hpp)
                const uint32_t s0 = (ring + 3) % kRingSlots;
                const uint32_t b = s0 / batch, o = s0 % batch, n1 = batch - o;
                const uint32_t page = first_row + g % gdn_copies;
                const uint32_t src = get_read_ptr(cb_hist_out);
                for (uint32_t t = 0; t < views_of(n_conv) * 32 / batch; t++) {
                    for (uint32_t f = 0; f < 2; f++) {
                        const uint32_t at = t * kTileBytes + f * batch * 32;
                        noc_async_write(src + at, hist.get_noc_addr(page, b * kBlkBytes + at + o * 32), n1 * 32);
                        if (o > 0) {
                            noc_async_write(
                                src + at + n1 * 32,
                                hist.get_noc_addr(page, ((b + 1) % kHistSlots) * kBlkBytes + at),
                                o * 32);
                        }
                    }
                }
                noc_async_write_barrier();
            } else if (rows_mode && preserve && n_conv > 0) {
                // the active rows' two face rows of each q|k|v tile of the 3 slots
                for (uint32_t ug = 0; ug < groups * batch; ug++) {
                    if (!((active_mask >> ug) & 1)) {
                        continue;
                    }
                    const uint32_t gr = ug / batch, u = ug % batch;
                    const uint32_t row = gr * hist_group_rows + first_row + g % gdn_copies;
                    for (uint32_t j = 0; j < outs; j++) {
                        for (uint32_t t = 0; t < n_conv; t++) {
                            for (uint32_t f = 0; f < 2; f++) {
                                const uint32_t off = t * kTileBytes + (f * batch + u) * 32;
                                noc_async_write(
                                    get_read_ptr(cb_hist_out) + (gr * outs + j) * kBlkViews * 2048 + off,
                                    hist.get_noc_addr(row, j * kBlkBytes + off),
                                    32);
                            }
                        }
                    }
                }
                noc_async_write_barrier();
            } else if (n_conv > 0) {
                // the 32x32 views holding the q|k|v tiles
                const uint32_t step = conv_step(ring, g, gdn_copies);
                for (uint32_t gr = 0; gr < groups; gr++) {
                    for (uint32_t j = 0; ag.has(gr) && j < outs; j++) {
                        noc_async_write(
                            get_read_ptr(cb_hist_out) + (gr * outs + j) * kBlkViews * 2048,
                            hist.get_noc_addr(
                                gr * hist_group_rows + first_row + g % gdn_copies,
                                (rows_mode ? j : (step + (verify_pair ? 2 + j : 0)) % kHistSlots) * kBlkBytes),
                            views_of(n_conv) * 2048);
                    }
                }
                noc_async_write_barrier();
            }
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_local)) = g + 1;
            cb_pop_front(cb_hist_out, groups * outs * kBlkViews);
            g++;
        }
        cb_pop_front(cb_qkvz_out, groups * kBlk);
        noc_semaphore_wait_min(heads_sem, o_expected);
        mark(l, 2);
        // (row groups: the mixers wrote group gr's o at gr * Ot tiles; o_tiles == Ot, see static_assert)
        cb_reserve_back(cb_o_in, kOinPages);
        cb_push_back(cb_o_in, kOinPages);
        send_partial(2 * l + kRound0);
        mark(l, 3);

        // mlp half
        take_slots(2 * l + 1 + kRound0);
        mark(l, 4);
        cb_wait_front(cb_aslice, groups * kBlk);
        mark(l, 5);
        // (row groups: group gr's slice at gr * kBlk tiles, to block gr of every streamer's act, It tiles per group)
        for (uint32_t gr = 0; gr < groups; gr++) {
            if (!ag.has(gr)) {
                continue;
            }
            const uint32_t slice = get_read_ptr(cb_aslice) + gr * kBlk * kTileBytes;
            for (uint32_t s = 0; s < num_streamers; s++) {
                const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
                const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
                noc_async_write(
                    slice, get_noc_addr(px, py, act_addr + (gr * It + act_off) * kTileBytes), ng * kTileBytes);
            }
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, act_sem_addr), 1);
        }
        cb_pop_front(cb_aslice, groups * kBlk);
        noc_semaphore_wait_min(act_sem, num_streamers * (l + 1));
        mark(l, 6);
        cb_reserve_back(cb_act, groups * It);
        cb_push_back(cb_act, groups * It);
        send_partial(2 * l + 1 + kRound0);
        mark(l, 7);
    }
    if constexpr (lm_head) {
        take_slots(2 * layers + kRound0);
        // (row groups: the records go to the hub's parity-0 slots, free once every sub-round is summed; argmax_addr is
        // then only the on / off flag)
        uint32_t rec_addr = argmax_addr;
        if constexpr (groups > 1) {
            while (hub_slots == 0) {
                invalidate_l1_cache();
                hub_slots = *hub_addr_sem;
            }
            rec_addr = hub_slots;
        }
        if (argmax_addr) {
            // bf16 bits -> unsigned key in the order of the values; per row, record slot core * kRecRows + row.
            // Column c of user u: tile c / 32, face (c % 32) / 16, face row u (row groups: group gr's logits at
            // gr * nv_max tiles; its user u is global row gr * batch + u)
            constexpr uint32_t kRecRows = groups * batch;  // the hub's kFeedRows
            cb_wait_front(cb_logits, groups * nv_max);
            const uint32_t base = get_read_ptr(cb_logits);
            auto key = [](uint32_t b) { return sampling::key_of(b); };
            auto word = [&](uint32_t gr, uint32_t u, uint32_t i) {
                const uint32_t c = i % 32;
                return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                    base + (gr * nv_max + i / 32) * kTileBytes + ((c / 16) * batch + u) * 32)[(c % 16) / 2];
            };
            volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_pout));
            if constexpr (sample) {
                // per row its list (rows mode: the active rows'; the others stay cleared), staged one group at a time;
                // row r's lists of every streamer back to back on the hub: [row][streamer]
                static_assert(nd_max * kTileBytes >= batch * sampling::kListBytes, "the lists are staged in cb_pout");
                for (uint32_t gr = 0; gr < groups; gr++) {
                    if (!ag.has(gr)) {
                        continue;  // (an inactive group's rows: the hub feeds none of them)
                    }
                    noc_async_write_barrier();  // the previous group's lists have left cb_pout
                    for (uint32_t u = 0; u < batch; u++) {
                        volatile tt_l1_ptr uint32_t* list = rec + u * 2 * sampling::kTopK;
                        sampling::clear(list);
                        if (rows_mode && !((active_mask >> (gr * batch + u)) & 1)) {
                            continue;
                        }
                        for (uint32_t i = 0; i < valid_cols; i += 2) {
                            const uint32_t w = word(gr, u, i);
                            sampling::insert(list, key(w & 0xffff), first_index + i);
                            if (i + 1 < valid_cols) {
                                sampling::insert(list, key(w >> 16), first_index + i + 1);
                            }
                        }
                    }
                    for (uint32_t u = 0; u < batch; u++) {
                        const uint32_t r = gr * batch + u;
                        noc_async_write(
                            get_write_ptr(cb_pout) + u * sampling::kListBytes,
                            get_noc_addr(hub_x, hub_y, rec_addr + (r * num_streamers + core) * sampling::kListBytes),
                            sampling::kListBytes);
                    }
                }
            } else {
                static_assert(nd_max * kTileBytes >= kRecRows * 16, "the records are staged in cb_pout");
                for (uint32_t r = 0; r < kRecRows; r++) {
                    const uint32_t gr = r / batch, u = r % batch;
                    uint32_t best = 0, best_i = 0;
                    for (uint32_t i = 0; ag.has(gr) && i < valid_cols; i += 2) {
                        const uint32_t w = word(gr, u, i);
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
                    rec[4 * r] = best;
                    rec[4 * r + 1] = first_index + best_i;
                }
                noc_async_write(
                    get_write_ptr(cb_pout), get_noc_addr(hub_x, hub_y, rec_addr + core * kRecRows * 16), kRecRows * 16);
            }
            noc_async_write_barrier();
            noc_semaphore_inc(hub_gather_sem, 1);  // the record is on the hub (a feeding hub waits for it)
        }
    }
    noc_async_atomic_barrier();
}
