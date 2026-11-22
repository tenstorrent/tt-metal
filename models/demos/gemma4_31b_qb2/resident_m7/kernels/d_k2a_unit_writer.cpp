// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K2a unit-core writer (BRISC, NoC 0: every read, write and atomic of this RISC; see k2a_common.hpp). Three queues,
// served in one loop so none blocks another (the K ring runs ahead across layers; partials and root hand-offs must
// still drain):
//   K: per layer of the core's units, per position tile, its Dt K tiles into the cb_k ring (a tail tile: read, the
//     user's new rows spliced in from the K staging once READY, written back to the cache, then pushed), then the
//     tile's mask into cb_mask where the tile is a window-start or tail tile;
//   partials: each unit's partial from cb_own into its root core's cb_parts slot, then + 1 on the root's semaphore;
//   roots: once a root's k partials are in (its semaphore), its pages of cb_parts to the compute; cb_parts padded to
//     NP after the layer's last root.
//
// Runtime args: 0 this unit core's index. Common: see d_k2a_unit_reader.cpp.
// d_k2a copy (M6b): drafter mode (kDraftW > 0) reads K from the slot-major ring (ring_tile0), no page tables.

#include "api/dataflow/dataflow_api.h"
#include "d_k2a_common.hpp"

using namespace k2a;

void kernel_main() {
    const uint32_t me = get_arg_val<uint32_t>(0);
    const uint32_t P = get_common_arg_val<uint32_t>(6);
    constexpr uint32_t gmap_arg = 7, pool_arg = 8;
    const uint32_t gmap = get_common_arg_val<uint32_t>(gmap_arg), pools = get_common_arg_val<uint32_t>(pool_arg);
    const uint32_t kv = get_common_arg_val<uint32_t>(9), k_stage = get_common_arg_val<uint32_t>(11);
    const uint32_t uxy = get_common_arg_val<uint32_t>(14);
    // this RISC's error words start clear (the hub reads them after the last layer)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_ERR_B)) = 0;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_ERR2_B)) = 0;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_ERRU_B)) = 0;
    // the schedule and the tables (the reader's)
    volatile tt_l1_ptr uint32_t* sched_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_SCHED));
    while (*sched_sem == 0) {
        invalidate_l1_cache();
    }
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* rec = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb_sched));
    volatile tt_l1_ptr uint32_t* users = rec + kScrUsers;
    const uint32_t nu = rec[0], nr = rec[1], nus = rec[2], nrs = rec[4];
    const uint32_t tab = kDraft ? 0 : cb_base(cb_tab), parts_base = cb_base(cb_parts), newk = cb_base(cb_newk);
    volatile tt_l1_ptr uint32_t* slots = rec + kFeedSlot;
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_READY));
    const InterleavedAddrGen<true> kst{.bank_base_address = k_stage, .page_size = kBf8};

    auto unit = [&](uint32_t i) { return UnitRec{rec + kUnitRec + kUW * i}; };
    auto root = [&](uint32_t r) { return RootRec{rec + kRootRec + kRW * r}; };
    auto tabs = [&](uint32_t i, uint32_t user) {
        return UnitTab{reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab + 4 * kTabW * i), P, gmap, pools, user,
                       U_ERR_B, U_ERR2_B, U_ERRU_B};
    };
    auto ulo = [&](uint32_t l) { return is_full(l) ? nus : 0; };
    auto uhi = [&](uint32_t l) { return is_full(l) ? nu : nus; };
    auto rlo = [&](uint32_t l) { return is_full(l) ? nrs : 0; };
    auto rhi = [&](uint32_t l) { return is_full(l) ? nr : nrs; };

    // queue states
    uint32_t kl = 0, ki = ulo(0), kt = 0, kpos = 0, kloaded = ~0u;  // K: layer, unit, local tile; ring position
    bool kpushed = false;
    uint32_t pl = 0, pi = ulo(0);           // partials: layer, unit
    uint32_t rl = 0, ri = rlo(0), rused = 0;  // roots: layer, root, pages handed over this layer
    for (;;) {
        bool busy = false;
        // ---- K (a unit index past the layer's range moves to the next layer)
        while (kl < layers && ki >= uhi(kl)) {
            kl++;
            ki = kl < layers ? ulo(kl) : 0;
            kt = 0;
        }
        if (kl < layers) {
            busy = true;
            const bool full = is_full(kl);
            const uint32_t Dt = full ? kDtf : kDts;
            const UnitRec u = unit(ki);
            const uint32_t n = u.n(), t = u.t0() + kt;
            const bool tail = u.tail() && kt + u.nT() >= n;
            bool go = true;
            if (tail && kloaded != kl * kMaxUnits + ki) {
                // the user's new K rows (staged by head 2g / head 0 once READY)
                if (*ready >= kHeads * (kl + 1)) {
                    const uint32_t gk = full ? 0 : u.g();
                    for (uint32_t d = 0; d < Dt; d++) {
                        read_rows(kst, gk * Dt + d, u.user(), newk + d * kBf8);
                    }
                    noc_async_read_barrier();
                    kloaded = kl * kMaxUnits + ki;
                } else {
                    go = false;
                    invalidate_l1_cache();
                }
            }
            if (go && !kpushed) {
                const uint32_t pad = (Dt - kpos % Dt) % Dt;
                if (pad && cb_pages_reservable_at_back(cb_k, pad)) {
                    cb_reserve_back(cb_k, pad);
                    cb_push_back(cb_k, pad);
                    kpos = (kpos + pad) % KR;
                }
                if (kpos % Dt == 0 && cb_pages_reservable_at_back(cb_k, Dt)) {
                    cb_reserve_back(cb_k, Dt);
                    const uint32_t wp = get_write_ptr(cb_k);
                    const InterleavedAddrGenFast<true> kg{
                        .bank_base_address = get_common_arg_val<uint32_t>(kv + 2 * kl), .page_size = kBf8,
                        .data_format = DataFormat::Bfp8_b};
                    const uint32_t id0 = kDraft ? ring_tile0(slots[u.user() * M], kl, u.g(), t)
                                                : tabs(ki, u.user()).kv_tile0(kl, full, t, full ? 0 : u.g());
                    for (uint32_t d = 0; d < Dt; d++) {
                        noc_async_read_page(id0 + d, kg, wp + d * kBf8);
                    }
                    noc_async_read_barrier();
                    if (tail) {
                        const uint32_t p = users[u.user()] & 0x7fffffff;
                        for (uint32_t d = 0; d < Dt; d++) {
                            splice_user(wp + d * kBf8, newk + d * kBf8, u.user(), p, u.tt0(), t - u.tt0());
                            noc_async_write_page(id0 + d, kg, wp + d * kBf8);
                        }
                        noc_async_write_barrier();
                    }
                    cb_push_back(cb_k, Dt);
                    kpos = (kpos + Dt) % KR;
                    kpushed = true;
                }
            }
            if (kpushed) {
                bool done = true;
                if (masked(u.nW(), u.nT(), n, kt)) {
                    done = false;
                    if (cb_pages_reservable_at_back(cb_mask, 1)) {
                        cb_reserve_back(cb_mask, 1);
                        const uint32_t p = users[u.user()] & 0x7fffffff;
                        if (full) {
                            build_mask<true>(get_write_ptr(cb_mask), t, u.user(), p);
                        } else {
                            build_mask<false>(get_write_ptr(cb_mask), t, u.user(), p);
                        }
                        cb_push_back(cb_mask, 1);
                        done = true;
                    }
                }
                if (done) {
                    kpushed = false;
                    if (++kt == n) {
                        kt = 0;
                        ki++;
                    }
                }
            }
        }
        // ---- partials
        while (pl < layers && pi >= uhi(pl)) {
            pl++;
            pi = pl < layers ? ulo(pl) : 0;
        }
        if (pl < layers) {
            busy = true;
            if (cb_pages_available_at_front(cb_own, kPT)) {
                const UnitRec u = unit(pi);
                const uint32_t PT = is_full(pl) ? kPTf : kPTs;
                const uint32_t xy = get_common_arg_val<uint32_t>(uxy + u.root());
                const uint64_t dst = get_noc_addr(xy & 0xffff, xy >> 16, parts_base + (u.pbase() + u.pslot() * PT) * kBf16);
                noc_async_write(get_read_ptr(cb_own), dst, PT * kBf16);
                noc_async_write_barrier();
                noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(U_PARTS + u.psem())), 1);
                noc_async_atomic_barrier();
                cb_pop_front(cb_own, kPT);
                pi++;
            }
        }
        // ---- roots (then the layer's pad)
        if (rl < layers) {
            busy = true;
            const bool full = is_full(rl);
            const uint32_t PT = full ? kPTf : kPTs;
            if (ri < rhi(rl)) {
                const RootRec r = root(ri);
                volatile tt_l1_ptr uint32_t* sem =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(U_PARTS + r.psem()));
                invalidate_l1_cache();
                if (*sem >= r.k() * layers_of_kind(full, rl) && cb_pages_reservable_at_back(cb_parts, r.k() * PT)) {
                    cb_reserve_back(cb_parts, r.k() * PT);
                    cb_push_back(cb_parts, r.k() * PT);
                    rused += r.k() * PT;
                    ri++;
                }
            } else {
                const uint32_t pad = NP - rused;
                if (pad == 0 || cb_pages_reservable_at_back(cb_parts, pad)) {
                    if (pad) {
                        cb_reserve_back(cb_parts, pad);
                        cb_push_back(cb_parts, pad);
                    }
                    rl++;
                    ri = rl < layers ? rlo(rl) : 0;
                    rused = 0;
                }
            }
        }
        if (!busy) {
            break;
        }
    }
    (void)me;
}
