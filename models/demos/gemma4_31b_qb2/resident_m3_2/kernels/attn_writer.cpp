// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident attention writer (BRISC, NOC 0), see attn_common.hpp. Per layer:
//   prep core (L_g sliding, L_0 full): once the compute has packed Q, multicast it into the workers' cb_q and bump
//     their A_Q; once its k children's partials are in (A_PARTS), hand them to the compute; the compute's final O
//     rows (32x32 tiles, rows h R + r) as R x 32 tiles into every streamer's o_in (multicast) and bump their sem_o.
//   worker: take Q (A_Q), then (active) send its partial (merger G_g with children: the merged one) into its
//     parent's cb_parts slot and bump the parent's A_PARTS.
// At launch: p into cb_pos for the compute; L_0 publishes the leaders' rx and parts addresses, G_0 the mergers' parts
// address (mailboxes: the streamers' sem_rxaddr, every attention core's A_PADDR_L / A_PADDR_G).
//
// Runtime args: 0 a, 1 g, 2 j, 3 full worker index (~0: none), 4 / 5 positions / slots (DRAM [1, 1, 1, 32] int32
// rows), 6 capacity, 7 max_users, 8 / 9 / 10 L_g / G_g / L_0 (x | y << 16), 11 parts pages of this core's cb_parts,
// 12 streamers' sem_o id, 13 streamers' sem_rxaddr id, 14 O dump (DRAM rows of 128 R x 32 tiles per layer: the
// o_proj input; 0 = off), 15 partial dump (DRAM pages of PTmax tiles, page l * 32 + a: this core's own partial;
// debug, 0 = off), 16 P (table columns), 17 .. 24 the layers' groups, 25 .. 30 pool pages per group, 31 .. 90 the
// layers' K cache addresses (M1L: workers stream their K tiles over NOC 0, the reader their V tiles over NOC 1),
// then rectangle lists
// (count, then x0, y0, x1, y1, cores each): Q targets of this core as a sliding leader, Q targets as the full
// leader, every other attention core, the streamers.

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace gatt;

namespace {

FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}

struct Rects {
    uint32_t arg, count;
    uint32_t v(uint32_t q, uint32_t i) const { return get_arg_val<uint32_t>(arg + 1 + 5 * q + i); }
    uint64_t addr(uint32_t q, uint32_t local) const { return get_noc_multicast_addr(v(q, 0), v(q, 1), v(q, 2), v(q, 3), local); }
    // bytes at src into local address dst of every core, then +1 on semaphore sem (0: none)
    void send(uint32_t src, uint32_t dst, uint32_t bytes, uint32_t sem) const {
        for (uint32_t q = 0; q < count; q++) {
            noc_async_write_multicast(src, addr(q, dst), bytes, v(q, 4));
        }
        noc_async_write_barrier();
        if (sem) {
            for (uint32_t q = 0; q < count; q++) {
                noc_semaphore_inc_multicast(addr(q, sem), 1, v(q, 4));
            }
            noc_async_atomic_barrier();
        }
    }
    void set(uint32_t word, uint32_t sem) const {
        for (uint32_t q = 0; q < count; q++) {
            noc_semaphore_set_multicast(word, addr(q, sem), v(q, 4));
        }
    }
};

FORCE_INLINE uint64_t xy_addr(uint32_t xy, uint32_t local) { return get_noc_addr(xy & 0xffff, xy >> 16, local); }

FORCE_INLINE uint32_t mailbox(uint32_t sem) {
    volatile tt_l1_ptr uint32_t* box = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem));
    uint32_t v = 0;
    while (v == 0) {
        invalidate_l1_cache();
        v = *box;
    }
    return v;
}

}  // namespace

void kernel_main() {
    const uint32_t a = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t j = get_arg_val<uint32_t>(2);
    const uint32_t wf = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4), slots_addr = get_arg_val<uint32_t>(5);
    const uint32_t capacity = get_arg_val<uint32_t>(6), max_users = get_arg_val<uint32_t>(7);
    const uint32_t lg_xy = get_arg_val<uint32_t>(8), gg_xy = get_arg_val<uint32_t>(9), l0_xy = get_arg_val<uint32_t>(10);
    const uint32_t parts_pages = get_arg_val<uint32_t>(11);
    const uint32_t sem_o = get_semaphore(get_arg_val<uint32_t>(12));
    const uint32_t sem_rxaddr = get_semaphore(get_arg_val<uint32_t>(13));
    const uint32_t odump = get_arg_val<uint32_t>(14);
    const uint32_t pdump = get_arg_val<uint32_t>(15);
    constexpr uint32_t PTmax = PTs > PTf ? PTs : PTf;
    const InterleavedAddrGen<true> pd{.bank_base_address = pdump, .page_size = kOwn * kBf16};
    const Tree tree{31 + layers};
    constexpr uint32_t kRects = 31 + layers + kTreeArgs;
    Rects q_s{kRects, get_arg_val<uint32_t>(kRects)};
    const uint32_t P = get_arg_val<uint32_t>(16);
    Rects q_f{q_s.arg + 1 + 5 * q_s.count, get_arg_val<uint32_t>(q_s.arg + 1 + 5 * q_s.count)};
    Rects q_fn{q_f.arg + 1 + 5 * q_f.count, get_arg_val<uint32_t>(q_f.arg + 1 + 5 * q_f.count)};  // narrow layers
    Rects others{q_fn.arg + 1 + 5 * q_fn.count, get_arg_val<uint32_t>(q_fn.arg + 1 + 5 * q_fn.count)};
    Rects streamers{others.arg + 1 + 5 * others.count, get_arg_val<uint32_t>(others.arg + 1 + 5 * others.count)};
    const bool leader = j == 0, merger = j == 1;

    // p for the compute (cb_pos word 0: the block as the reader sees it, launch_of; an inert or flagged launch runs
    // p = 0); words 4 and 8 stage the published addresses (16 B aligned) once p is taken
    cb_reserve_back(cb_pos, 1);
    const uint32_t pos = get_write_ptr(cb_pos);
    bool inert = true;
    {
        const InterleavedAddrGen<true> pg{.bank_base_address = pos_addr, .page_size = 128};
        const InterleavedAddrGen<true> sg{.bank_base_address = slots_addr, .page_size = 128};
        noc_async_read(pg.get_noc_addr(0), pos, 64);
        noc_async_read(sg.get_noc_addr(0), pos + 64, 64);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos);
        const Launch L = launch_of(w, w + 16, capacity, max_users);
        w[0] = L.p;
        inert = L.inert;
    }
    const Plan pl = make_plan(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos)[0]);
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos);
    if (a == 0) {
        stage[4] = cb_base(cb_parts);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_PADDR_L)) = stage[4];
        others.set(pos + 16, get_semaphore(A_PADDR_L));
        stage[8] = cb_base(cb_rx);
        streamers.set(pos + 32, sem_rxaddr);
        noc_async_write_barrier();
    } else if (g == 0 && merger) {
        stage[4] = cb_base(cb_parts);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_PADDR_G)) = stage[4];
        others.set(pos + 16, get_semaphore(A_PADDR_G));
        noc_async_write_barrier();
    }
    cb_push_back(cb_pos, 1);

    volatile tt_l1_ptr uint32_t* q_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_Q));
    volatile tt_l1_ptr uint32_t* parts_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_PARTS));
    uint32_t q_count = 0, parts_count = 0;
    uint32_t o_in = 0;
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t Dt = full ? kDtf : kDts, QT = full ? QTf : QTs, PT = full ? PTf : PTs;
        if (g >= kGroups && !(full && pl.wide)) {
            continue;  // M1L full-only cores: wide full layers only
        }
        if (full ? a == 0 : leader) {
            // ---- root: Q to the workers
            cb_wait_front(cb_qready, 1);
            cb_pop_front(cb_qready, 1);
            (full ? (pl.wide ? q_f : q_fn) : q_s).send(cb_base(cb_q), cb_base(cb_q), QT * Dt * kBf16, get_semaphore(A_Q));
            // the children's partials to the compute
            const uint32_t k = full ? tree.active_children(pl) : pl.s_a;
            parts_count += k;
            noc_semaphore_wait_min(parts_sem, parts_count);
            cb_reserve_back(cb_parts, parts_pages);
            cb_push_back(cb_parts, parts_pages);
            // O rows (h R + r of 32x32 tiles) as R x 32 tiles into every streamer's o_in
            const uint32_t heads = full ? 8 : 2;
            cb_wait_front(cb_out, kQ32);
            const uint32_t src = get_read_ptr(cb_out), ost = cb_base(cb_ostage);
            for (uint32_t h = 0; h < heads; h++) {
                const uint32_t i0 = h * R;
                for (uint32_t d = 0; d < Dt; d++) {
                    const uint32_t tile = src + ((i0 / 32) * Dt + d) * kBf16 + ((i0 % 32) / 16) * 1024 + (i0 % 16) * 32;
                    for (uint32_t f = 0; f < 2; f++) {
                        noc_async_write(
                            tile + f * 512, get_noc_addr(ost + (h * Dt + d) * kRowTile + f * 32 * R), 32 * R);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, kQ32);
            if (pdump) {
                noc_async_write(cb_base(cb_own), pd.get_noc_addr(l * 32 + a), kOwn * kBf16);
                noc_async_write_barrier();
            }
            if (o_in == 0) {
                o_in = mailbox(A_OADDR);
            }
            const uint32_t first = full ? 0 : g * 2 * kDts;
            streamers.send(ost, o_in + first * kRowTile, heads * Dt * kRowTile, sem_o);
            if (odump) {
                const InterleavedAddrGen<true> od{.bank_base_address = odump, .page_size = 128 * kRowTile};
                noc_async_write(ost, od.get_noc_addr(l, first * kRowTile), heads * Dt * kRowTile);
                noc_async_write_barrier();
            }
        } else if (!leader) {
            // ---- worker: wide full layers: its K tiles (M1L: this RISC on NOC 0; the first kPreK of them before Q
            // lands, as the reader starts V); then Q, then this core's partial to its parent
            {
                uint32_t t0 = 0, n = 0;
                bool act;
                if (full) {
                    act = wf < pl.f_a;
                    if (act) {
                        tree.split_full(pl, wf, t0, n);
                    }
                } else {
                    act = j - 1 < pl.s_a;
                    if (act) {
                        split(pl.s_n, pl.s_a, j - 1, t0, n);
                        t0 += pl.s_lo;
                    }
                }
                const bool kw = act && full && pl.wide;  // (narrow layers: the reader streams K into cb_k2)
                if (kw) {
                    volatile tt_l1_ptr uint32_t* ready =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_PTAB));
                    while (*ready == 0) {
                        invalidate_l1_cache();
                    }
                    const Tables tabs{reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb_ptab)), P, 17, 25,
                                      inert};
                    const InterleavedAddrGenFast<true> kgen{
                        .bank_base_address = get_arg_val<uint32_t>(31 + l), .page_size = kBf8,
                        .data_format = DataFormat::Bfp8_b};
                    const uint32_t head = full ? 0 : g;
                    for (uint32_t t = t0; t < t0 + n; t++) {
                        if (t == t0 + kPreK) {
                            noc_semaphore_wait_min(q_sem, ++q_count);
                            cb_reserve_back(cb_q, CQ);
                            cb_push_back(cb_q, CQ);
                        }
                        cb_reserve_back(cb_k, kKW);
                        const uint32_t wp = get_write_ptr(cb_k);
                        for (uint32_t d = 0; d < Dt; d++) {
                            noc_async_read_page(tabs.kv_tile(l, full, t, head, d), kgen, wp + d * kBf8);
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_k, kKW);
                    }
                }
                if (!kw || n <= kPreK) {
                    noc_semaphore_wait_min(q_sem, ++q_count);
                    cb_reserve_back(cb_q, CQ);
                    cb_push_back(cb_q, CQ);
                }
            }
            bool active;
            uint32_t parent, slot, box;
            if (full) {
                active = wf < pl.f_a;
                parent = tree.parent_xy();
                slot = tree.slot();
                box = tree.box();
                const uint32_t k = tree.merger() ? tree.active_children(pl) : 0;
                if (active && k > 0) {
                    parts_count += k;
                    noc_semaphore_wait_min(parts_sem, parts_count);
                    cb_reserve_back(cb_parts, parts_pages);
                    cb_push_back(cb_parts, parts_pages);
                }
            } else {
                active = j - 1 < pl.s_a;
                parent = lg_xy;
                slot = j - 1;
                box = A_PADDR_L;
            }
            if (active) {
                // (a merger always sends cb_pout: its compute forwards or merges every partial)
                const bool m = full ? tree.merger() : merger;
                const uint32_t cb = m ? cb_pout : cb_own;
                const uint32_t pages = m ? PTf : kOwn;
                cb_wait_front(cb, pages);
                const uint32_t dst = mailbox(box) + slot * PT * kBf16;
                noc_async_write(get_read_ptr(cb), xy_addr(parent, dst), PT * kBf16);
                noc_async_write_barrier();
                noc_semaphore_inc(xy_addr(parent, get_semaphore(A_PARTS)), 1);
                noc_async_atomic_barrier();
                if (pdump) {
                    noc_async_write(cb_base(cb_own), pd.get_noc_addr(l * 32 + a), kOwn * kBf16);
                    noc_async_write_barrier();
                }
                cb_pop_front(cb, pages);
            }
        }
    }
    (void)merger;
}
