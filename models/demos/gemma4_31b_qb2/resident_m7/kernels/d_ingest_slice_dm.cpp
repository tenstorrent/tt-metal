// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter ingest, slice (BRISC; M6b C1), see d_ingest_common.hpp. K = 8 slices, slice k: hidden
// tiles [k T, (k + 1) T) of ctx, the K-tile range [qa, qa + nq) of the fc input, and head block k (k head k for
// k < 4, v head k - 4 otherwise). Per launch (rows 0 .. M - 1, R = 32 rows of tiles):
//   in    for each live row and tap, this slice's columns of the taps from the lazy rows into 32 x 32 tiles (face
//         layout; inert rows zero), multicast into every streamer's cb_in at tile qa; their sem_in + 1. The lazy rows
//         (DRAM, ROW_MAJOR bf16 [slots x 5 taps x 32, 1344], one 2,688 B row each: chip c's 1344 columns of the tap):
//         row (slot * 5 + tap) * 32 + position mod 32, written by the target's slices after each tapped layer
//   fc    the all-reduce round's slots of every chip (the streamers' fc partials, the peer hubs' packets) to the compute
//   ctx   the hidden norm's exchange (as d_slice_dm.cpp); ctx multicast into every streamer's cb_h; sem_in + 1
//   per drafter layer: the k | v tiles of head block k from its 4 streamers (sem_kv) to the compute; the compute's
//         BF8 tiles (k: k_norm + RoPE; v: as is) into the ring: per ring page the launch's rows land on, read the
//         page's 8 tiles, replace those rows (each row's shared exponents and mantissas of a BF8 tile are its own
//         bytes), write the page back
// The ring (slot-major, M6A addendum 1 §4.5): per (slot, layer, kv head) 72 pages of 32 positions, position p at page
// (p / 32) mod 72, row p mod 32; a page is 8 BF8 32 x 32 tiles (256 columns): DRAM TILE tensor [*, 256], tile
// (((slot * layers_ring + layer) * 4 + head) * 72 + page) * 8 + column tile. K and V are separate tensors.
//
// Compile-time args: 0 num_chips, 1 chip, 2 k, 3 K, 4 T, 5 V, 6 layers, 7 local streamers, 8 P, 9 sem_in (the
// streamers'), 10 sem_exch, 11 sem_kv, 12 M, 13 qa, 14 nq, 15 ring pages (72), 16 ring layers (5), 17 slots per ring
// Runtime args: 0 sem_haddr id, 1 slot semaphore (parity 0), 2 launch counter word, 3 lazy taps, 4 hidden norm weight
// (DRAM rows of K x V views), 5 positions (DRAM [1, 32] int32: row r's position, -1 = none), 6 slots (DRAM [1, 32]),
// 7 RoPE (DRAM TILE [64, 256] bf16: tiles 0..7 cos, 8..15 sin of the launch's rows), 8 k_norm weights (DRAM TILE
// [layers x 32, 256] bf16, row-replicated: tile l * 8 + c), 9 ring (K or V), 10 pack dump (DRAM TILE BF8 [K x layers x
// 32, 256]: tile (k layers + l) * 8 + c, 0 = off), 11 ctx dump (DRAM rows of K x V views, 0 = off), 12 sem_slotaddr id,
// 13 sem_kvaddr id, 14 hub x | y << 16, 15 rectangles of the streamers, then per rectangle x0, y0, x1, y1, cores,
// then K x (x, y) of the slices.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t chip = get_compile_time_arg_val(1);
constexpr uint32_t k = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3);
constexpr uint32_t T = get_compile_time_arg_val(4);
constexpr uint32_t V = get_compile_time_arg_val(5);
constexpr uint32_t layers = get_compile_time_arg_val(6);
constexpr uint32_t n_loc = get_compile_time_arg_val(7);
constexpr uint32_t P = get_compile_time_arg_val(8);
constexpr uint32_t sem_in = get_compile_time_arg_val(9);
constexpr uint32_t sem_exch = get_compile_time_arg_val(10);
constexpr uint32_t sem_kv = get_compile_time_arg_val(11);
constexpr uint32_t M = get_compile_time_arg_val(12);
constexpr uint32_t qa = get_compile_time_arg_val(13);
constexpr uint32_t nq = get_compile_time_arg_val(14);
constexpr uint32_t ring_pages = get_compile_time_arg_val(15);
constexpr uint32_t ring_layers = get_compile_time_arg_val(16);
constexpr uint32_t ring_slots = get_compile_time_arg_val(17);
static_assert(K > 1 && M <= 32, "");
constexpr bool is_k = k < 4;
constexpr uint32_t head = k % 4;

constexpr uint32_t cb_slots = 0, cb_hout = 4, cb_const = 6, cb_pss = 7, cb_exch = 8, cb_wpre = 11;
constexpr uint32_t cb_kv = 12, cb_wk = 13, cb_rope = 14, cb_out8 = 16, cb_in = 17, cb_scr = 18;
constexpr uint32_t kView = 2048, kF32Tile = 4096, kBf8Tile = 1088, kHeadTiles = 8;
constexpr uint32_t kTapCols = 1344, kTapTiles = 42, kRowBytes = kTapCols * 2;
constexpr uint32_t slice_bytes = T * kView;
constexpr uint32_t slot_bytes = V * kView;
constexpr uint32_t per_round = n_loc + (num_chips - 1) * P;

// byte offset of row r's 16 elements of face column half f in a 32 x 32 tile (bf16: 32 B per face row)
FORCE_INLINE uint32_t face_row(uint32_t r, uint32_t f) { return (r / 16) * 1024 + f * 512 + (r % 16) * 32; }

void kernel_main() {
    size_t a = 0;
    volatile tt_l1_ptr uint32_t* haddr_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(a++)));
    const uint32_t slot_sem_addr = get_arg_val<uint32_t>(a++);
    const uint32_t launch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lazy_addr = get_arg_val<uint32_t>(a++);
    const uint32_t wnorm_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slots_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rope_addr = get_arg_val<uint32_t>(a++);
    const uint32_t knorm_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ring_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pdump_addr = get_arg_val<uint32_t>(a++);
    const uint32_t cdump_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slotaddr_box = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t kvaddr_box = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t hub_xy = get_arg_val<uint32_t>(a++);
    const uint32_t rects = get_arg_val<uint32_t>(a++);
    const size_t rect_args = a;
    a += 5 * rects;
    const size_t slice_args = a;
    auto rect = [&](uint32_t q, uint32_t i) { return get_arg_val<uint32_t>(rect_args + 5 * q + i); };
    auto cb_addr = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    const uint32_t slots = cb_addr(cb_slots), exch_addr = cb_addr(cb_exch);
    // scratch: [0, 128) positions, [128, 256) slots, [256, 256 + row) one lazy row segment, then one ring page
    const uint32_t scr = cb_addr(cb_scr);
    const uint32_t row_buf = scr + 256, page_buf = scr + 256 + kRowBytes;
    if constexpr (k == 0) {
        const uint32_t word = get_write_ptr(cb_const);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = slots;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16) = cb_addr(cb_kv);
        noc_semaphore_set_remote(word, get_noc_addr(hub_xy & 0xffff, hub_xy >> 16, slotaddr_box));
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_set_multicast(
                word, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), slotaddr_box), rect(q, 4));
            noc_semaphore_set_multicast(
                word + 16, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), kvaddr_box),
                rect(q, 4));
        }
        noc_async_write_barrier();
    }
    // the norms' constant tiles: eye, ones, fold (R = 32: the eye)
    cb_reserve_back(cb_const, 3);
    {
        volatile tt_l1_ptr uint16_t* c = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_const));
        for (uint32_t i = 0; i < 1024; i++) {
            const uint32_t face = i / 256, row = (face / 2) * 16 + (i % 256) / 16, col = (face % 2) * 16 + i % 16;
            c[i] = row == col ? 0x3f80 : 0;
            c[1024 + i] = 0x3f80;
            c[2048 + i] = row == col ? 0x3f80 : 0;
        }
    }
    cb_push_back(cb_const, 3);

    const InterleavedAddrGen<true> lazy{.bank_base_address = lazy_addr, .page_size = kRowBytes};
    const InterleavedAddrGen<true> wnorm{.bank_base_address = wnorm_addr, .page_size = K * slot_bytes};
    const InterleavedAddrGen<true> rowg{.bank_base_address = pos_addr, .page_size = 128};
    const InterleavedAddrGen<true> slotg{.bank_base_address = slots_addr, .page_size = 128};
    const InterleavedAddrGen<true> ropeg{.bank_base_address = rope_addr, .page_size = kView};
    const InterleavedAddrGen<true> knormg{.bank_base_address = knorm_addr, .page_size = kView};
    const InterleavedAddrGen<true> ringg{.bank_base_address = ring_addr, .page_size = kBf8Tile};
    const InterleavedAddrGen<true> pdumpg{.bank_base_address = pdump_addr, .page_size = kBf8Tile};
    const InterleavedAddrGen<true> cdumpg{.bank_base_address = cdump_addr, .page_size = K * slot_bytes};

    // the hidden norm weight, the rows' positions / slots, RoPE (k heads)
    cb_reserve_back(cb_wpre, V);
    noc_async_read(wnorm.get_noc_addr(0, k * slot_bytes), get_write_ptr(cb_wpre), slot_bytes);
    noc_async_read(rowg.get_noc_addr(0), scr, 128);
    noc_async_read(slotg.get_noc_addr(0), scr + 128, 128);
    if constexpr (is_k) {
        cb_reserve_back(cb_rope, 2 * kHeadTiles);
        for (uint32_t t = 0; t < 2 * kHeadTiles; t++) {
            noc_async_read(ropeg.get_noc_addr(t), get_write_ptr(cb_rope) + t * kView, kView);
        }
    }
    noc_async_read_barrier();
    cb_push_back(cb_wpre, V);
    if constexpr (is_k) {
        cb_push_back(cb_rope, 2 * kHeadTiles);
    }
    volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(scr);
    volatile tt_l1_ptr uint32_t* slot_of = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 128);

    // in: this slice's K tiles [qa, qa + nq) of the taps, tile q = column tile q % 42 of tap q / 42
    cb_reserve_back(cb_in, nq);
    const uint32_t in_base = get_write_ptr(cb_in);
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base);
        for (uint32_t i = 0; i < nq * kView / 4; i++) {
            z[i] = 0;
        }
    }
    for (uint32_t t = qa / kTapTiles; t <= (qa + nq - 1) / kTapTiles; t++) {
        const uint32_t q0 = qa > t * kTapTiles ? qa : t * kTapTiles;
        const uint32_t q1 = qa + nq < (t + 1) * kTapTiles ? qa + nq : (t + 1) * kTapTiles;
        const uint32_t c0 = q0 - t * kTapTiles;
        for (uint32_t r = 0; r < M; r++) {
            if (pos[r] < 0 || slot_of[r] >= ring_slots) {
                continue;  // an inert row stays zero
            }
            const uint32_t lrow = (slot_of[r] * 5 + t) * 32 + static_cast<uint32_t>(pos[r]) % 32;
            noc_async_read(lazy.get_noc_addr(lrow, c0 * 64), row_buf, (q1 - q0) * 64);
            noc_async_read_barrier();
            for (uint32_t q = q0; q < q1; q++) {
                for (uint32_t f = 0; f < 2; f++) {
                    volatile tt_l1_ptr uint32_t* d =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_base + (q - qa) * kView + face_row(r, f));
                    volatile tt_l1_ptr uint32_t* s =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row_buf + (q - q0) * 64 + f * 32);
                    for (uint32_t w = 0; w < 8; w++) {
                        d[w] = s[w];
                    }
                }
            }
        }
    }
    uint32_t h_dst = 0;
    while (h_dst == 0) {
        invalidate_l1_cache();
        h_dst = *haddr_box;
    }
    auto mcast = [&](uint32_t src, uint32_t dst, uint32_t bytes) {
        for (uint32_t q = 0; q < rects; q++) {
            noc_async_write_multicast(
                src, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), dst), bytes, rect(q, 4));
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_inc_multicast(
                get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), get_semaphore(sem_in)),
                1,
                rect(q, 4));
        }
        noc_async_atomic_barrier();
    };
    mcast(in_base, h_dst + qa * kView, nq * kView);
    cb_push_back(cb_in, nq);  // (no consumer: the CB is this RISC's staging)

    // fc: the all-reduce round (parity 0; the launch counter bases the monotonic slot semaphore)
    volatile tt_l1_ptr uint32_t* launch_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(launch_addr);
    invalidate_l1_cache();
    const uint32_t launch = *launch_word;
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_sem_addr), (launch + 1) * per_round);
    cb_reserve_back(cb_slots, num_chips * V);
    cb_push_back(cb_slots, num_chips * V);

    // ctx: the hidden norm's exchange, then multicast
    {
        cb_wait_front(cb_pss, 1);
        const uint32_t src = get_read_ptr(cb_pss);
        const uint32_t dst = exch_addr + k * kF32Tile;
        for (uint32_t j = 0; j < K; j++) {
            noc_async_write(
                src,
                get_noc_addr(get_arg_val<uint32_t>(slice_args + 2 * j), get_arg_val<uint32_t>(slice_args + 2 * j + 1), dst),
                kF32Tile);
        }
        noc_async_write_barrier();
        for (uint32_t j = 0; j < K; j++) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(slice_args + 2 * j),
                    get_arg_val<uint32_t>(slice_args + 2 * j + 1),
                    get_semaphore(sem_exch)),
                1);
        }
        cb_pop_front(cb_pss, 1);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_exch)), K);
        cb_reserve_back(cb_exch, K);
        cb_push_back(cb_exch, K);
    }
    cb_wait_front(cb_hout, V);
    mcast(get_read_ptr(cb_hout), h_dst + k * slice_bytes, slice_bytes);
    if (cdump_addr) {
        noc_async_write(get_read_ptr(cb_hout), cdumpg.get_noc_addr(0, k * slot_bytes), slot_bytes);
        noc_async_write_barrier();
    }
    cb_pop_front(cb_hout, V);

    // per layer: head block k's k | v tiles to the compute, its BF8 tiles into the ring
    volatile tt_l1_ptr uint32_t* kv_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_kv));
    for (uint32_t l = 0; l < layers; l++) {
        if constexpr (is_k) {
            cb_reserve_back(cb_wk, kHeadTiles);
            for (uint32_t c = 0; c < kHeadTiles; c++) {
                noc_async_read(knormg.get_noc_addr(l * kHeadTiles + c), get_write_ptr(cb_wk) + c * kView, kView);
            }
            noc_async_read_barrier();
            cb_push_back(cb_wk, kHeadTiles);
        }
        noc_semaphore_wait_min(kv_sem, 4 * (l + 1));
        cb_reserve_back(cb_kv, kHeadTiles);
        cb_push_back(cb_kv, kHeadTiles);
        cb_wait_front(cb_out8, kHeadTiles);
        const uint32_t src = get_read_ptr(cb_out8);
        if (pdump_addr) {
            for (uint32_t c = 0; c < kHeadTiles; c++) {
                noc_async_write(src + c * kBf8Tile, pdumpg.get_noc_addr((k * layers + l) * kHeadTiles + c), kBf8Tile);
            }
            noc_async_write_barrier();
        }
        // ring read-modify-write, page by page (the first row of each page gathers that page's rows)
        for (uint32_t r = 0; r < M; r++) {
            if (pos[r] < 0) {
                continue;
            }
            const uint32_t pg = (static_cast<uint32_t>(pos[r]) / 32) % ring_pages;
            const uint32_t sl = slot_of[r];
            bool first = true;
            for (uint32_t r2 = 0; r2 < r && first; r2++) {
                first = !(pos[r2] >= 0 && slot_of[r2] == sl && (static_cast<uint32_t>(pos[r2]) / 32) % ring_pages == pg);
            }
            if (!first || sl >= ring_slots) {
                continue;
            }
            const uint32_t tile0 = (((sl * ring_layers + l) * 4 + head) * ring_pages + pg) * kHeadTiles;
            for (uint32_t c = 0; c < kHeadTiles; c++) {
                noc_async_read(ringg.get_noc_addr(tile0 + c), page_buf + c * kBf8Tile, kBf8Tile);
            }
            noc_async_read_barrier();
            for (uint32_t r2 = r; r2 < M; r2++) {
                if (pos[r2] < 0 || slot_of[r2] != sl || (static_cast<uint32_t>(pos[r2]) / 32) % ring_pages != pg) {
                    continue;
                }
                const uint32_t d = static_cast<uint32_t>(pos[r2]) % 32;
                for (uint32_t c = 0; c < kHeadTiles; c++) {
                    volatile tt_l1_ptr uint8_t* st = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src + c * kBf8Tile);
                    volatile tt_l1_ptr uint8_t* dt = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(page_buf + c * kBf8Tile);
                    for (uint32_t f = 0; f < 2; f++) {
                        const uint32_t fs = (r2 / 16) * 2 + f, fd = (d / 16) * 2 + f;
                        const uint32_t es = fs * 16 + r2 % 16, ed = fd * 16 + d % 16;
                        dt[ed] = st[es];
                        volatile tt_l1_ptr uint32_t* ms = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(st + 64 + es * 16);
                        volatile tt_l1_ptr uint32_t* md = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dt + 64 + ed * 16);
                        for (uint32_t w = 0; w < 4; w++) {
                            md[w] = ms[w];
                        }
                    }
                }
            }
            for (uint32_t c = 0; c < kHeadTiles; c++) {
                noc_async_write(page_buf + c * kBf8Tile, ringg.get_noc_addr(tile0 + c), kBf8Tile);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(cb_out8, kHeadTiles);
    }
    *launch_word = launch + 1;
}
