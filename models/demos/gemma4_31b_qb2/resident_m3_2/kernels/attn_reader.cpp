// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident attention reader (NCRISC), see attn_common.hpp. At launch: the block (row 0's position and slot, see launch_of) and the slot's tables, the constant
// tiles (ones, identity), the window masks of tiles lo and lo + 1, the page tables; on a leader the tail masks, the
// cos / sin tiles of rows p + i % R (from the ttnn decode RoPE tables) and zeroed raw Q / KV tiles. Per layer:
//   prep (L_g sliding, L_0 full): the layer's q / k norm weights and the tail cache tiles; once every streamer's
//     q|k|v rows are in rx, their R x 32 tiles laid into 32x32 tiles (Q rows h R + r, K / V rows r) for the compute;
//     the compute's new BF8 K / V rows spliced into the tail tiles (rows (p + r) % 32: one read-modify-write per tile,
//     so the M rows of one tile never race), the tiles to the compute and back to the cache.
//   worker: this core's position tiles of the layer: K tiles per position tile, then V tiles per d (all positions),
//     running ahead of the compute across layers as far as the CBs allow.
//
// Runtime args: 0 a (core index), 1 g, 2 j (index in the group), 3 full-layer worker index (~0: none),
// 4 positions, 5 slots (DRAM [1, 1, 1, 32] int32 rows), 6 tables (DRAM [users, 6, P] int32), 7 scratch pages (DRAM
// [users, 1, 16] int32: group g at word g), 8 P (table columns), 9 capacity, 10 max_users, 11 / 12 cos / sin sliding, 13 / 14 full (DRAM
// row-major [positions, D] bf16), 15 norm weight tiles (DRAM bf16, tile (2 l + k) * 16 + d), 16 .. 23 the layers'
// groups (4 bits each), 24 .. 29 pool pages per group, then per layer the K and V cache addresses.

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace gatt;

namespace {

FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}

// a bf16 32x32 tile at addr from value(i, c) (row i, column c), two elements per word
template <typename F>
FORCE_INLINE void fill_tile(uint32_t addr, F value) {
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t face = 0; face < 4; face++) {
        for (uint32_t ri = 0; ri < 16; ri++) {
            const uint32_t i = (face / 2) * 16 + ri;
            for (uint32_t ci = 0; ci < 16; ci += 2) {
                const uint32_t c = (face % 2) * 16 + ci;
                w[(face * 256 + ri * 16 + ci) / 2] = value(i, c) | (static_cast<uint32_t>(value(i, c + 1)) << 16);
            }
        }
    }
}

FORCE_INLINE void copy_words(uint32_t dst, uint32_t src, uint32_t words) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < words; i++) {
        d[i] = s[i];
    }
}

// rows p + i % R of a [positions, Dt * 32] bf16 table as Dt 32x32 tiles at dst (scratch: R rows)
FORCE_INLINE void rope_tiles(uint32_t table, uint32_t Dt, uint32_t p, uint32_t dst, uint32_t scratch) {
    const uint32_t row_bytes = Dt * 64;
    const InterleavedAddrGen<true> gen{.bank_base_address = table, .page_size = row_bytes};
    for (uint32_t r = 0; r < R; r++) {
        noc_async_read(gen.get_noc_addr(p + r), scratch + r * row_bytes, row_bytes);
    }
    noc_async_read_barrier();
    for (uint32_t d = 0; d < Dt; d++) {
        for (uint32_t i = 0; i < 32; i++) {
            for (uint32_t f = 0; f < 2; f++) {
                copy_words(
                    dst + d * kBf16 + (i / 16) * 1024 + f * 512 + (i % 16) * 32,
                    scratch + (i % R) * row_bytes + (d * 32 + f * 16) * 2,
                    8);
            }
        }
    }
}

// F1 (M2b): rows [r0, r0 + n) of the BF8 tile at src (all in face rows 0 .. 15, as the compute packs R <= 8 new
// rows) into rows [rd0, rd0 + n) of the BF8 tile at dst (one face-row half: rd0 % 16 + n <= 16): per face of the
// pair one exponent run and one mantissa run instead of two face-row copies per row (byte-identical result)
FORCE_INLINE void splice_rows(uint32_t dst, uint32_t rd0, uint32_t src, uint32_t r0, uint32_t n) {
    volatile tt_l1_ptr uint8_t* de = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst);
    volatile tt_l1_ptr uint8_t* se = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
    const uint32_t hd = (rd0 / 16) * 2, od = rd0 % 16;
    for (uint32_t f = 0; f < 2; f++) {
        for (uint32_t i = 0; i < n; i++) {
            de[(hd + f) * 16 + od + i] = se[f * 16 + r0 + i];
        }
        copy_words(dst + 64 + (hd + f) * 256 + od * 16, src + 64 + f * 256 + r0 * 16, 4 * n);
    }
}

}  // namespace

void kernel_main() {
    const uint32_t a = get_arg_val<uint32_t>(0);
    const uint32_t g = get_arg_val<uint32_t>(1);
    const uint32_t j = get_arg_val<uint32_t>(2);
    const uint32_t wf = get_arg_val<uint32_t>(3);
    const uint32_t pos_addr = get_arg_val<uint32_t>(4), slots_addr = get_arg_val<uint32_t>(5);
    const uint32_t tab_addr = get_arg_val<uint32_t>(6), scr_addr = get_arg_val<uint32_t>(7);
    const uint32_t P = get_arg_val<uint32_t>(8), capacity = get_arg_val<uint32_t>(9);
    const uint32_t max_users = get_arg_val<uint32_t>(10);
    const uint32_t cos_s = get_arg_val<uint32_t>(11), sin_s = get_arg_val<uint32_t>(12);
    const uint32_t cos_f = get_arg_val<uint32_t>(13), sin_f = get_arg_val<uint32_t>(14);
    const uint32_t w_addr = get_arg_val<uint32_t>(15);
    constexpr uint32_t gmap_arg = 16, pool_arg = 24, kv_args = 30;
    const Tree tree{kv_args + 2 * layers};  // (M1L: the full-layer split's skew)
    const bool leader = j == 0;

    // this launch's table errors start clear (the hub reads them after the last layer)
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_ERR)) = 0;
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_ERR2)) = 0;
    // the block (row 0's position and slot), the slot's scratch pages and per-group tables, the plan
    const uint32_t ptab = cb_base(cb_ptab);
    volatile tt_l1_ptr uint32_t* pt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ptab);
    {
        const InterleavedAddrGen<true> pos{.bank_base_address = pos_addr, .page_size = 128};
        const InterleavedAddrGen<true> sl{.bank_base_address = slots_addr, .page_size = 128};
        noc_async_read(pos.get_noc_addr(0), ptab + 4 * kPosW, 128);
        noc_async_read(sl.get_noc_addr(0), ptab + 4 * kSlotW, 128);
        noc_async_read_barrier();
    }
    const Launch L = launch_of(pt + kPosW, pt + kSlotW, capacity, max_users);
    {
        const uint32_t slot = L.inert ? 0 : L.slot;  // an out-of-range slot uses slot 0's scratch pages
        const InterleavedAddrGen<true> scr{.bank_base_address = scr_addr, .page_size = 64};
        noc_async_read(scr.get_noc_addr(slot), ptab + 4 * kScratchW, 64);
        if (!L.inert) {
            const InterleavedAddrGen<true> tab{.bank_base_address = tab_addr, .page_size = 4 * P};
            for (uint32_t grp = 0; grp < kGroupsMax; grp++) {
                noc_async_read(tab.get_noc_addr(slot * kGroupsMax + grp), ptab + 4 * (kTabBase + grp * P), 4 * P);
            }
        }
        noc_async_read_barrier();
    }
    const Tables tabs{pt, P, gmap_arg, pool_arg, L.inert};
    const Plan pl = make_plan(L.p);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_PTAB)) = 1;  // for the writer's K stream

    // constants: ones, identity; window masks of tiles lo, lo + 1 (row i sees positions >= p + i % R - 1023)
    cb_reserve_back(cb_one, 2);
    fill_tile(get_write_ptr(cb_one), [](uint32_t, uint32_t) -> uint16_t { return 0x3f80; });
    fill_tile(get_write_ptr(cb_one) + kBf16, [](uint32_t i, uint32_t c) -> uint16_t { return i == c ? 0x3f80 : 0; });
    cb_push_back(cb_one, 2);
    cb_reserve_back(cb_wmask, 2);
    for (uint32_t m = 0; m < 2; m++) {
        const uint32_t t = pl.s_lo + m;
        fill_tile(get_write_ptr(cb_wmask) + m * kBf16, [&](uint32_t i, uint32_t c) -> uint16_t {
            return 32 * t + c + (kWindow - 1) < pl.p + i % R ? kNegBig : 0;
        });
    }
    cb_push_back(cb_wmask, 2);

    if (leader) {
        // tail masks (row i sees positions <= p + i % R)
        cb_reserve_back(cb_tmask, 2);
        for (uint32_t m = 0; m < 2; m++) {
            const uint32_t t = pl.tt0 + m;
            fill_tile(get_write_ptr(cb_tmask) + m * kBf16, [&](uint32_t i, uint32_t c) -> uint16_t {
                return 32 * t + c > pl.p + i % R ? kNegBig : 0;
            });
        }
        cb_push_back(cb_tmask, 2);
        // cos / sin tiles (the ostage buffer is the scratch: nothing writes it before the first layer's output)
        cb_reserve_back(cb_cs, kCS);
        const uint32_t cs = get_write_ptr(cb_cs), scratch = cb_base(cb_ostage);
        rope_tiles(cos_s, kDts, pl.p, cs, scratch);
        rope_tiles(sin_s, kDts, pl.p, cs + kDts * kBf16, scratch);
        if (a == 0) {
            rope_tiles(cos_f, kDtf, pl.p, cs + 2 * kDts * kBf16, scratch);
            rope_tiles(sin_f, kDtf, pl.p, cs + (2 * kDts + kDtf) * kBf16, scratch);
        }
        cb_push_back(cb_cs, kCS);
        // raw Q / KV tiles: rows nothing writes stay zero
        const uint32_t zero_cbs[2] = {cb_q32, cb_kv32};
        for (uint32_t cb : zero_cbs) {
            auto& iface = get_local_cb_interface(cb);
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(cb));
            for (uint32_t i = 0; i < iface.fifo_size / 4; i++) {
                z[i] = 0;
            }
        }
    }

    volatile tt_l1_ptr uint32_t* rx_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_RX));
    const uint32_t rx = cb_base(cb_rx);
    const uint32_t my_noc_x = my_x[noc_index], my_noc_y = my_y[noc_index];
    const InterleavedAddrGenFast<true> wgen{
        .bank_base_address = w_addr, .page_size = kBf16, .data_format = DataFormat::Float16_b};
    uint32_t preps = 0;
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t Dt = full ? kDtf : kDts;
        if (g >= kGroups && !(full && pl.wide)) {
            continue;  // M1L full-only cores: wide full layers only
        }
        const uint32_t head = full ? 0 : g;
        const InterleavedAddrGenFast<true> kgen{
            .bank_base_address = get_arg_val<uint32_t>(kv_args + 2 * l), .page_size = kBf8, .data_format = DataFormat::Bfp8_b};
        const InterleavedAddrGenFast<true> vgen{
            .bank_base_address = get_arg_val<uint32_t>(kv_args + 2 * l + 1), .page_size = kBf8, .data_format = DataFormat::Bfp8_b};
        if (full ? a == 0 : leader) {
            // ---- prep
            cb_reserve_back(cb_w, 32);
            for (uint32_t d = 0; d < Dt; d++) {
                noc_async_read_page((2 * l) * 16 + d, wgen, get_write_ptr(cb_w) + d * kBf16);
                noc_async_read_page((2 * l + 1) * 16 + d, wgen, get_write_ptr(cb_w) + (16 + d) * kBf16);
            }
            noc_async_write_barrier();  // the previous prep's write-back of the tail tiles
            cb_reserve_back(cb_tk, kTail);
            cb_reserve_back(cb_tv, kTail);
            const uint32_t tk = get_write_ptr(cb_tk), tv = get_write_ptr(cb_tv);
            for (uint32_t c = 0; c < pl.ct; c++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    const uint32_t id = tabs.kv_tile(l, full, pl.tt0 + c, head, d);
                    noc_async_read_page(id, kgen, tk + (c * Dt + d) * kBf8);
                    noc_async_read_page(id, vgen, tv + (c * Dt + d) * kBf8);
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_w, 32);

            // q|k|v rows: R x 32 tile c of the group's rx -> 32x32 tile rows
            noc_semaphore_wait_min(rx_sem, num_streamers * ++preps);
            cb_reserve_back(cb_q32, kQ32);
            cb_reserve_back(cb_kv32, 16);
            const uint32_t q32 = get_write_ptr(cb_q32), kv32 = get_write_ptr(cb_kv32);
            const uint32_t nrx = full ? 128 + kDtf : 16 + 2 * kDts;  // full 144, sliding 32
            for (uint32_t c = 0; c < nrx; c++) {
                uint32_t dst, i0;
                if (full) {
                    if (c < 128) {
                        i0 = (c / 16) * R;
                        dst = q32 + ((i0 / 32) * kDtf + c % 16) * kBf16;
                    } else {
                        i0 = 0;
                        dst = kv32 + (c - 128) * kBf16;
                    }
                } else {
                    if (c < 16) {
                        i0 = (c / 8) * R;
                        dst = q32 + ((i0 / 32) * kDts + c % 8) * kBf16;
                    } else {
                        i0 = 0;
                        dst = kv32 + (c - 16) * kBf16;  // K at [0, 8), V at [8, 16)
                    }
                }
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(
                        rx + c * kRowTile + f * 32 * R,
                        get_noc_addr(my_noc_x, my_noc_y, dst + ((i0 % 32) / 16) * 1024 + f * 512 + (i0 % 16) * 32),
                        32 * R);
                }
            }
            noc_async_write_barrier();
            cb_push_back(cb_q32, kQ32);
            cb_push_back(cb_kv32, 16);

            // the compute's new K / V rows into the tail tiles, which go to the compute and back to the cache
            cb_wait_front(cb_k8, 16);
            cb_wait_front(cb_v8, 16);
            const uint32_t k8 = get_read_ptr(cb_k8), v8 = get_read_ptr(cb_v8);
            // F1: the M new rows as runs that share a tail tile and a face-row half (at most 2 runs for M <= 8)
            for (uint32_t r0 = 0; r0 < M;) {
                const uint32_t pos = pl.p + r0, c = pos / 32 - pl.tt0, rd0 = pos % 32;
                const uint32_t room = 16 - rd0 % 16, n = M - r0 < room ? M - r0 : room;
                for (uint32_t d = 0; d < Dt; d++) {
                    splice_rows(tk + (c * Dt + d) * kBf8, rd0, k8 + d * kBf8, r0, n);
                    splice_rows(tv + (c * Dt + d) * kBf8, rd0, v8 + d * kBf8, r0, n);
                }
                r0 += n;
            }
            cb_pop_front(cb_k8, 16);
            cb_pop_front(cb_v8, 16);
            cb_push_back(cb_tk, kTail);
            cb_push_back(cb_tv, kTail);
            for (uint32_t c = 0; c < pl.ct; c++) {
                for (uint32_t d = 0; d < Dt; d++) {
                    const uint32_t id = tabs.kv_tile(l, full, pl.tt0 + c, head, d);
                    noc_async_write_page(id, kgen, tk + (c * Dt + d) * kBf8);
                    noc_async_write_page(id, vgen, tv + (c * Dt + d) * kBf8);
                }
            }
        } else if (!leader) {
            // ---- worker: this core's position tiles
            uint32_t t0 = 0, n = 0;
            bool active;
            if (full) {
                active = wf < pl.f_a;
                if (active) {
                    tree.split_full(pl, wf, t0, n);
                }
            } else {
                active = j - 1 < pl.s_a;
                if (active) {
                    split(pl.s_n, pl.s_a, j - 1, t0, n);
                    t0 += pl.s_lo;
                }
            }
            if (!active) {
                continue;
            }
            const uint32_t NV = full ? NVf : NVs;
            const bool kr = !(full && pl.wide);  // narrow layers: this RISC reads K too (wide: the writer, NOC 0)
            // chunks of NV position tiles (M1L; sliding: one chunk): [K per position tile,] then V per d
            for (uint32_t c0 = t0; c0 < t0 + n; c0 += NV) {
                const uint32_t c1 = c0 + NV < t0 + n ? c0 + NV : t0 + n;
                for (uint32_t t = c0; kr && t < c1; t++) {
                    cb_reserve_back(cb_k2, kKW);
                    const uint32_t wp = get_write_ptr(cb_k2);
                    for (uint32_t d = 0; d < Dt; d++) {
                        noc_async_read_page(tabs.kv_tile(l, full, t, head, d), kgen, wp + d * kBf8);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_k2, kKW);
                }
                for (uint32_t d = 0; d < Dt; d++) {
                    cb_reserve_back(cb_v, NV);
                    const uint32_t wp = get_write_ptr(cb_v);
                    for (uint32_t t = c0; t < c1; t++) {
                        noc_async_read_page(tabs.kv_tile(l, full, t, head, d), vgen, wp + (t - c0) * kBf8);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_v, NV);
                }
                // pad cb_v back to its base (the next chunk's / layer's windows may differ)
                const uint32_t pad = (CV - (Dt * NV) % CV) % CV;
                if (pad) {
                    cb_reserve_back(cb_v, pad);
                    cb_push_back(cb_v, pad);
                }
            }
        }
    }
    noc_async_write_barrier();
}
