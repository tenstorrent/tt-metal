// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K2a unit-core reader (NCRISC, NoC 1: every read, write and atomic of this RISC; see k2a_common.hpp).
// At launch: the feed rows, the users, the schedule (Builder, identical on every unit core) into cb_sched, every unit's
// page-table entries into cb_tab; then the writer (U_SCHED) and the compute (cb_sched) go, with the constants.
// Then three queues, served in one loop so none blocks another (the V ring runs ahead across layers):
//   V: per layer of the core's units, per sub-chunk, per d, the sub-chunk's V tiles into the cb_v ring (NV pages; a
//     tail tile: read, the user's new rows spliced in from the V staging once READY, written back to the cache);
//   Q: once READY (every head core's staging of the layer), the layer's Q keys from the Q staging into cb_q, one
//      key's slot per push (K2b step 1: a 2-slot ring; the compute's units come sorted by key);
//   O: each root's output (cb_out) as its user's rows of each q head into that head core's cb_og, then + 1 on its
//     H_OG.
//
// Runtime args: 0 this unit core's index. Common: 0 positions, 1 slots (DRAM [1, 1, 1, 32] int32 rows), 2 capacity,
// 3 max_users, 4 tables (DRAM [users, 6, P] int32), 5 scratch pages (DRAM [users, 1, 16] int32), 6 P (table columns),
// 7 offset of the 8 packed layer -> group words, 8 offset of the kGroupsMax pool page counts, 9 offset of the layers'
// K, V cache addresses, 10 Q staging, 11 K staging, 12 V staging, 13 head core xy offset (kHeads), 14 unit core xy
// offset (kNumUnits)

#include "api/dataflow/dataflow_api.h"
#include "k2a_common.hpp"

using namespace k2a;

namespace {
}  // namespace

void kernel_main() {
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t capacity = get_common_arg_val<uint32_t>(2), max_users = get_common_arg_val<uint32_t>(3);
    const uint32_t tab_addr = get_common_arg_val<uint32_t>(4), scr_addr = get_common_arg_val<uint32_t>(5);
    const uint32_t P = get_common_arg_val<uint32_t>(6);
    const uint32_t gmap = get_common_arg_val<uint32_t>(7), pools = get_common_arg_val<uint32_t>(8);
    const uint32_t kv = get_common_arg_val<uint32_t>(9), q_stage = get_common_arg_val<uint32_t>(10);
    const uint32_t v_stage = get_common_arg_val<uint32_t>(12);
    const uint32_t hxy = get_common_arg_val<uint32_t>(13);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_ERR_N)) = 0;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_ERR2_N)) = 0;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_ERRU_N)) = 0;

    // ---- launch: feed rows, users, schedule
    const uint32_t sched = cb_base(cb_sched);
    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sched);
    {
        const InterleavedAddrGen<true> pg{.bank_base_address = get_common_arg_val<uint32_t>(0), .page_size = 128};
        const InterleavedAddrGen<true> sg{.bank_base_address = get_common_arg_val<uint32_t>(1), .page_size = 128};
        noc_async_read(pg.get_noc_addr(0), sched + 4 * kFeedPos, 128);
        noc_async_read(sg.get_noc_addr(0), sched + 4 * kFeedSlot, 128);
        noc_async_read_barrier();
    }
    volatile tt_l1_ptr uint32_t* users = rec + kScrUsers;
    volatile tt_l1_ptr uint32_t* slots = rec + kFeedSlot;
    for (uint32_t u = 0; u < U; u++) {
        const User L = user_of(rec + kFeedPos, slots, capacity, max_users, u);
        users[u] = L.live ? (L.p | 0x80000000u) : 0;
    }
    const bool fit = Builder{rec, me}.build();
    {
        // (K2b K1) this core's schedule records (the cb_sched page) to DRAM row `me` for the host replica check
        const InterleavedAddrGen<true> sd{.bank_base_address = get_common_arg_val<uint32_t>(16), .page_size = 4096};
        noc_async_write(sched, sd.get_noc_addr(me), 4096);
        noc_async_write_barrier();
    }
    if (me == 0) {
        // (K2b step 3) the schedule status (global: every unit core built the same fit) to every head core, which
        // waits for it once per launch before layer 0; on overflow also this core's U_SCHERR word (the hub: F_SCHED)
        if (!fit) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_SCHERR)) = 1;
        }
        for (uint32_t h = 0; h < kHeads; h++) {
            const uint32_t xy = get_common_arg_val<uint32_t>(hxy + h);
            noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(H_SCHED)), fit ? kSchedOk : kSchedOverflow);
        }
        noc_async_atomic_barrier();
    }
    const uint32_t nu = rec[0], nr = rec[1], nus = rec[2], nrs = rec[4], nqs = rec[5], nqf = rec[6];

    // ---- every unit's table entries (its user's slot: the scratch page, then per used group of its kind the 64 B
    // granules holding its columns, staged in cb_newv, which nothing uses before the first tail splice)
    const uint32_t tab = cb_base(cb_tab), tmp = (cb_base(cb_newv) + 63) & ~63u;
    {
        const InterleavedAddrGen<true> scr{.bank_base_address = scr_addr, .page_size = 64};
        const InterleavedAddrGen<true> tg{.bank_base_address = tab_addr, .page_size = 4 * P};
        for (uint32_t i = 0; i < nu; i++) {
            const UnitRec u{rec + kUnitRec + kUW * i};
            const uint32_t slot = slots[u.user() * M];
            const uint32_t region = tab + 4 * kTabW * i;
            volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(region);
            const uint32_t lo = u.t0() / 4, nc = (u.t0() + u.n() - 1) / 4 - lo + 1;
            const uint32_t hi = lo + nc < P ? lo + nc : P;
            noc_async_read(scr.get_noc_addr(slot), region, 64);
            uint32_t off = tmp;
            for (uint32_t pass = 0; pass < 2; pass++) {
                off = tmp;
                for (uint32_t g = 0; g < kGroupsMax; g++) {
                    if (!group_used(g) || group_full(g) != u.full() || lo >= hi) {
                        continue;
                    }
                    const uint32_t k0 = lo / kGranule, k1 = (hi - 1) / kGranule;
                    if (pass == 0) {
                        for (uint32_t k = k0; k <= k1; k++) {
                            noc_async_read(tg.get_noc_addr(slot * kGroupsMax + g, 64 * k), off + 64 * (k - k0), 64);
                        }
                    } else {
                        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(off);
                        const uint32_t e0 = 16 + kind_rank(g) * (u.full() ? kColsF : kColsS);
                        for (uint32_t c = 0; c < hi - lo; c++) {
                            w[e0 + c] = src[lo % kGranule + c];
                        }
                    }
                    off += 64 * (k1 - k0 + 1);
                }
                if (pass == 0) {
                    noc_async_read_barrier();
                }
            }
            w[8] = lo;
            w[9] = nc;
        }
    }
    // the writer and the compute go; constants: ones (the reduce scaler), identity (the merges' diag)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_SCHED)) = 1;
    cb_reserve_back(cb_sched, 1);
    cb_push_back(cb_sched, 1);
    cb_reserve_back(cb_one, 2);
    {
        volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_one));
        for (uint32_t i = 0; i < kBf16 / 4; i++) {
            o[i] = 0x3f803f80;
        }
        volatile tt_l1_ptr uint16_t* id = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_one) + kBf16);
        for (uint32_t face = 0; face < 4; face++) {
            for (uint32_t ri = 0; ri < 16; ri++) {
                for (uint32_t ci = 0; ci < 16; ci++) {
                    const uint32_t i = (face / 2) * 16 + ri, c = (face % 2) * 16 + ci;
                    id[face * 256 + ri * 16 + ci] = i == c ? 0x3f80 : 0;
                }
            }
        }
    }
    cb_push_back(cb_one, 2);

    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_READY));
    volatile tt_l1_ptr uint32_t* ogaddr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_OGADDR));
    const uint32_t newv = cb_base(cb_newv);
    const InterleavedAddrGen<true> vst{.bank_base_address = v_stage, .page_size = kBf8};
    const InterleavedAddrGenFast<true> qst{
        .bank_base_address = q_stage, .page_size = kBf16, .data_format = DataFormat::Float16_b};
    auto unit = [&](uint32_t i) { return UnitRec{rec + kUnitRec + kUW * i}; };
    auto ulo = [&](uint32_t l) { return is_full(l) ? nus : 0; };
    auto uhi = [&](uint32_t l) { return is_full(l) ? nu : nus; };
    auto rlo = [&](uint32_t l) { return is_full(l) ? nrs : 0; };
    auto rhi = [&](uint32_t l) { return is_full(l) ? nr : nrs; };

    // queue states
    uint32_t vl = 0, vi = ulo(0), vj = 0, vd = 0, vpos = 0, vloaded = ~0u;  // V: layer, unit, sub-chunk, d; ring
    uint32_t ids[NVs];  // the sub-chunk's tile ids (d = 0), looked up once per sub-chunk
    bool have_ids = false;
    uint32_t ql = 0, qq = 0;       // Q: layer, next unit of the layer's kind (K2b: one slot per run of equal keys)
    uint32_t qpos = 0;             // (K2b) Q ring pages pushed (the resident layout starts at a multiple of 2 kQSlot)
    uint32_t ol = 0, oi = rlo(0);  // O: layer, root
    for (;;) {
        bool busy = false;
        // ---- Q
        // (K2b) Q into the 2-slot ring of kQSlot pages. A kind with <= 2 keys on this core: resident, every key once
        // (slot q at q kQSlot) in one push from a ring position aligned to 0 (a 16-page pad first when needed).
        // More keys: one push per run of equal keys in the compute's unit order ((tail, key) sorted). A layer with no
        // units of its kind pushes nothing (READY is still marked)
        if (ql < layers) {
            busy = true;
            invalidate_l1_cache();
            const bool full = is_full(ql);
            const uint32_t q0 = full ? nqs : 0, u0 = ulo(ql), u1 = uhi(ql), nq = full ? nqf : nqs;
            if (nq <= 2 && *ready >= kHeads * (ql + 1)) {
                if (nq > 0 && qpos % (2 * kQSlot) != 0) {
                    if (cb_pages_reservable_at_back(cb_q, kQSlot)) {  // the pad (the compute pops it alike)
                        cb_reserve_back(cb_q, kQSlot);
                        cb_push_back(cb_q, kQSlot);
                        qpos += kQSlot;
                    }
                } else if (nq == 0 || cb_pages_reservable_at_back(cb_q, nq * kQSlot)) {
                    K2A_MARK(ql, 0, kTlUnit);
                    if (nq > 0) {
                        const uint32_t Dt = full ? kDtf : kDts;
                        const uint32_t tiles = full ? Pack<true>::tiles : Pack<false>::tiles;
                        cb_reserve_back(cb_q, nq * kQSlot);
                        const uint32_t wp = get_write_ptr(cb_q);
                        for (uint32_t q = 0; q < nq; q++) {
                            const uint32_t key = rec[kQRec + q0 + q];
                            const uint32_t T = full ? key : (key >> 8) * tiles + (key & 0xff);
                            for (uint32_t d = 0; d < Dt; d++) {
                                noc_async_read_page(T * Dt + d, qst, wp + (q * kQSlot + d) * kBf16);
                            }
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_q, nq * kQSlot);
                        qpos += nq * kQSlot;
                    }
                    ql++;
                    qq = 0;
                }
            } else if (nq > 2 && *ready >= kHeads * (ql + 1) && (u0 + qq >= u1 || cb_pages_reservable_at_back(cb_q, kQSlot))) {
                if (qq == 0) {
                    K2A_MARK(ql, 0, kTlUnit);
                }
                if (u0 + qq < u1) {
                    const uint32_t kq = unit(u0 + qq).qslot();
                    const uint32_t Dt = full ? kDtf : kDts;
                    cb_reserve_back(cb_q, kQSlot);
                    const uint32_t wp = get_write_ptr(cb_q);
                    const uint32_t tiles = full ? Pack<true>::tiles : Pack<false>::tiles;
                    const uint32_t key = rec[kQRec + q0 + kq];
                    const uint32_t T = full ? key : (key >> 8) * tiles + (key & 0xff);
                    for (uint32_t d = 0; d < Dt; d++) {
                        noc_async_read_page(T * Dt + d, qst, wp + d * kBf16);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_q, kQSlot);
                    qpos += kQSlot;
                    while (u0 + qq < u1 && unit(u0 + qq).qslot() == kq) {  // the run of this key
                        qq++;
                    }
                }
                if (u0 + qq >= u1) {
                    ql++;
                    qq = 0;
                }
            }
        }
        // ---- V
        while (vl < layers && vi >= uhi(vl)) {
            vl++;
            vi = vl < layers ? ulo(vl) : 0;
            vj = vd = 0;
            have_ids = false;
        }
        if (vl < layers) {
            busy = true;
            const bool full = is_full(vl);
            const uint32_t Dt = full ? kDtf : kDts, NV = full ? NVf : NVs;
            const UnitRec u = unit(vi);
            uint32_t s0, sn;
            subchunk(full, u.t0(), u.n(), vj, s0, sn);
            // does this sub-chunk hold tail tiles (the unit's last nT)?
            const uint32_t tail0 = u.t0() + u.n() - (u.tail() ? u.nT() : 0);
            const bool has_tail = u.tail() && s0 + sn > tail0;
            bool go = true;
            if (has_tail && vloaded != vl * kMaxUnits + vi) {
                invalidate_l1_cache();
                if (*ready >= kHeads * (vl + 1)) {
                    const uint32_t gk = full ? 0 : u.g();
                    for (uint32_t d = 0; d < Dt; d++) {
                        read_rows(vst, gk * Dt + d, u.user(), newv + d * kBf8);
                    }
                    noc_async_read_barrier();
                    vloaded = vl * kMaxUnits + vi;
                } else {
                    go = false;
                }
            }
            if (go) {
                const uint32_t pad = (NV - vpos % NV) % NV;
                if (pad && cb_pages_reservable_at_back(cb_v, pad)) {
                    cb_reserve_back(cb_v, pad);
                    cb_push_back(cb_v, pad);
                    vpos = (vpos + pad) % VR;
                }
                if (vpos % NV == 0 && cb_pages_reservable_at_back(cb_v, NV)) {
                    const UnitTab tb{reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab + 4 * kTabW * vi), P, gmap,
                                     pools, u.user(), U_ERR_N, U_ERR2_N, U_ERRU_N};
                    if (!have_ids) {
                        for (uint32_t t = 0; t < sn; t++) {
                            ids[t] = tb.kv_tile0(vl, full, s0 + t, full ? 0 : u.g());
                        }
                        have_ids = true;
                    }
                    const InterleavedAddrGenFast<true> vg{
                        .bank_base_address = get_common_arg_val<uint32_t>(kv + 2 * vl + 1), .page_size = kBf8,
                        .data_format = DataFormat::Bfp8_b};
                    cb_reserve_back(cb_v, NV);
                    const uint32_t wp = get_write_ptr(cb_v);
                    for (uint32_t t = 0; t < sn; t++) {
                        noc_async_read_page(ids[t] + vd, vg, wp + t * kBf8);
                    }
                    noc_async_read_barrier();
                    if (has_tail) {
                        const uint32_t p = users[u.user()] & 0x7fffffff;
                        for (uint32_t t = 0; t < sn; t++) {
                            if (s0 + t >= tail0) {
                                splice_user(wp + t * kBf8, newv + vd * kBf8, u.user(), p, u.tt0(), s0 + t - u.tt0());
                                noc_async_write_page(ids[t] + vd, vg, wp + t * kBf8);
                            }
                        }
                        noc_async_write_barrier();
                    }
                    cb_push_back(cb_v, NV);
                    vpos = (vpos + NV) % VR;
                    if (++vd == Dt) {
                        vd = 0;
                        have_ids = false;
                        if (++vj == subchunks(full, u.t0(), u.n())) {
                            vj = 0;
                            vi++;
                        }
                    }
                }
            }
        }
        // ---- O
        while (ol < layers && oi >= rhi(ol)) {
            ol++;
            oi = ol < layers ? rlo(ol) : 0;
        }
        if (ol < layers) {
            busy = true;
            if (cb_pages_available_at_front(cb_out, 16)) {
                const bool full = is_full(ol);
                const uint32_t Dt = full ? kDtf : kDts;
                const RootRec r{rec + kRootRec + kRW * oi};
                const uint32_t uu = r.user();
                uint32_t og = *ogaddr;
                while (og == 0) {
                    invalidate_l1_cache();
                    og = *ogaddr;
                }
                const uint32_t src = get_read_ptr(cb_out);
                const uint32_t Ht = full ? Pack<true>::Ht : Pack<false>::Ht;
                for (uint32_t hl = 0; hl < Ht; hl++) {
                    const uint32_t h = full ? r.g() * Ht + hl : 2 * r.g() + hl;  // (full: g is the half)
                    const uint32_t sr = full ? Pack<true>::row(uu, h, 0) : Pack<false>::row(uu, hl, 0);
                    const uint32_t dr = uu * M;
                    const uint32_t xy = get_common_arg_val<uint32_t>(hxy + h);
                    for (uint32_t d = 0; d < Dt; d++) {
                        for (uint32_t f = 0; f < 2; f++) {
                            noc_async_write(src + d * kBf16 + (sr / 16) * 1024 + f * 512 + (sr % 16) * 32,
                                            get_noc_addr(xy & 0xffff, xy >> 16,
                                                         og + d * kBf16 + (dr / 16) * 1024 + f * 512 + (dr % 16) * 32),
                                            32 * M);
                        }
                    }
                }
                noc_async_write_barrier();
                for (uint32_t hl = 0; hl < Ht; hl++) {
                    const uint32_t h = full ? r.g() * Ht + hl : 2 * r.g() + hl;
                    const uint32_t xy = get_common_arg_val<uint32_t>(hxy + h);
                    noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(H_OG)), 1);
                }
                noc_async_atomic_barrier();
                K2A_MARK(ol, 1, kTlUnit);
                cb_pop_front(cb_out, 16);
                oi++;
            }
        }
        if (!busy) {
            break;
        }
    }
}
