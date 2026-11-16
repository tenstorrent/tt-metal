// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// K2a head-core reader (NCRISC, NoC 1; see k2a_common.hpp). At launch: the feed rows, the live users, the cos / sin
// tiles of every launch row's position (both kinds), the reduce scaler; head 0 publishes rx's address to the streamers.
// Per layer: the q (and k) norm weights; once every streamer's q|k|v columns are in rx (A_RX), rx to the compute; once
// every live user's O rows of this q head are in cb_og (H_OG), its Dt tiles into every streamer's o_in (one multicast
// per streamer rectangle, NoC 1), then +1 on their sem_o.
//
// Runtime args: 0 h. Common: 0 positions, 1 slots (DRAM [1, 1, 1, 32] int32 rows), 2 capacity, 3 max_users,
// 4 / 5 cos / sin sliding, 6 / 7 full (DRAM row-major [positions, D] bf16), 8 norm weight tiles (DRAM bf16, tile
// (2 l + k) * 16 + d), 9 streamers' sem_o id, 10 streamers' sem_rxaddr id, 11 streamer rectangle list offset, 12 streamer
// xy list offset (head 0's rx publish), 13 unit core xy list offset (head 0 publishes cb_og's address to them); at the
// offsets: (count, then x0, y0, x1, y1, cores each), (x | y << 16 each)

#include "api/dataflow/dataflow_api.h"
#include "k2a_common.hpp"

using namespace k2a;

namespace {

// rows pos[r] (r < 32) of a [positions, Dt * 32] bf16 table as Dt 32x32 tiles at dst (scratch: 32 rows)
void rope_tiles(uint32_t table, uint32_t Dt, volatile tt_l1_ptr uint32_t* rows, uint32_t dst, uint32_t scratch) {
    const uint32_t row_bytes = Dt * 64;
    const InterleavedAddrGen<true> gen{.bank_base_address = table, .page_size = row_bytes};
    for (uint32_t r = 0; r < 32; r++) {
        noc_async_read(gen.get_noc_addr(rows[r]), scratch + r * row_bytes, row_bytes);
    }
    noc_async_read_barrier();
    for (uint32_t d = 0; d < Dt; d++) {
        for (uint32_t i = 0; i < 32; i++) {
            for (uint32_t f = 0; f < 2; f++) {
                copy_words(dst + d * kBf16 + (i / 16) * 1024 + f * 512 + (i % 16) * 32,
                           scratch + i * row_bytes + (d * 32 + f * 16) * 2, 8);
            }
        }
    }
}

}  // namespace

void kernel_main() {
    const uint32_t h = get_arg_val<uint32_t>(0);
    const uint32_t pos_addr = get_common_arg_val<uint32_t>(0), slots_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t capacity = get_common_arg_val<uint32_t>(2), max_users = get_common_arg_val<uint32_t>(3);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t sem_o_id = get_common_arg_val<uint32_t>(9), rxaddr_id = get_common_arg_val<uint32_t>(10);
    const uint32_t rects = get_common_arg_val<uint32_t>(11), sxy = get_common_arg_val<uint32_t>(12);

    // feed rows -> the live users, every launch row's RoPE row (an inert or flagged user's rows: 0)
    const uint32_t feed = cb_base(hcb_feed);
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(feed);
    volatile tt_l1_ptr uint32_t* slot = pos + 32;
    volatile tt_l1_ptr uint32_t* rows = pos + 64;
    {
        const InterleavedAddrGen<true> pg{.bank_base_address = pos_addr, .page_size = 128};
        const InterleavedAddrGen<true> sg{.bank_base_address = slots_addr, .page_size = 128};
        noc_async_read(pg.get_noc_addr(0), feed, 128);
        noc_async_read(sg.get_noc_addr(0), feed + 128, 128);
        noc_async_read_barrier();
    }
    for (uint32_t r = 0; r < 32; r++) {  // (D16: rope_tiles reads 32 rows; at R = 8 rows >= 8 are padding)
        rows[r] = 0;
    }
    uint32_t live = 0;
    for (uint32_t u = 0; u < U; u++) {
        const User L = user_of(pos, slot, capacity, max_users, u);
        live += L.live ? 1 : 0;
        for (uint32_t m = 0; m < M; m++) {
            rows[u * M + m] = L.live ? L.p + m : 0;
        }
    }
    // (K2b step 3) the schedule status from unit core 0 is read once per launch, just before layer 0's O gather (the
    // first use of `live`): on overflow every unit core runs inert, so no root sends O rows: gather none (the hub
    // flags F_SCHED). Waiting before layer 0's prep instead cost +0.14 ms per launch (the heads idled on the Builder)
    volatile tt_l1_ptr uint32_t* hsched = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(H_SCHED));

    // constants: the reduce scaler; cos / sin (cb_og is the scratch: no root writes it before this core's first READY)
    cb_reserve_back(hcb_one, 1);
    {
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(hcb_one));
        for (uint32_t i = 0; i < kBf16 / 4; i++) {
            w[i] = 0x3f803f80;
        }
    }
    cb_push_back(hcb_one, 1);
    cb_reserve_back(hcb_cs, kCSPages);
    const uint32_t cs = get_write_ptr(hcb_cs), scratch = cb_base(hcb_og);
    rope_tiles(get_common_arg_val<uint32_t>(4), kDts, rows, cs, scratch);
    rope_tiles(get_common_arg_val<uint32_t>(5), kDts, rows, cs + kDts * kBf16, scratch);
    rope_tiles(get_common_arg_val<uint32_t>(6), kDtf, rows, cs + 2 * kDts * kBf16, scratch);
    rope_tiles(get_common_arg_val<uint32_t>(7), kDtf, rows, cs + (2 * kDts + kDtf) * kBf16, scratch);
    cb_push_back(hcb_cs, kCSPages);

    // head 0: rx's address (the same on every head core) to every streamer, cb_og's to every unit core
    if (h == 0) {
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t xy = get_common_arg_val<uint32_t>(sxy + s);
            noc_inline_dw_write(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(rxaddr_id)),
                                R == 32 ? cb_base(hcb_rx) : cb_base(hcb_rx8));  // (D16: R = 8 tiles land in rx8)
        }
        const uint32_t uxy = get_common_arg_val<uint32_t>(13);
        for (uint32_t c = 0; c < kNumUnits; c++) {
            const uint32_t xy = get_common_arg_val<uint32_t>(uxy + c);
            noc_inline_dw_write(get_noc_addr(xy & 0xffff, xy >> 16, get_semaphore(U_OGADDR)), cb_base(hcb_og));
        }
    }

    volatile tt_l1_ptr uint32_t* rx_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(H_RX));
    volatile tt_l1_ptr uint32_t* og_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(H_OG));
    volatile tt_l1_ptr uint32_t* oaddr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(H_OADDR));
    const InterleavedAddrGenFast<true> wgen{
        .bank_base_address = w_addr, .page_size = kBf16, .data_format = DataFormat::Float16_b};
    const uint32_t og = cb_base(hcb_og);
    const uint32_t nrect = get_common_arg_val<uint32_t>(rects);
    // (M5 D16) R = 8: local copies (NoC reads of this core's own L1) pad the streamers' 8-row tiles into hcb_rx's
    // 32-row tiles (faces 0 / 1 rows 0 .. 7; the rest stays zero) and repack O's 32-row tiles into 8-row tiles
    const uint64_t me = get_noc_addr(my_x[noc_index], my_y[noc_index], 0);
    auto local = [&](uint32_t src, uint32_t dst, uint32_t bytes) { noc_async_read(me | src, dst, bytes); };
    if constexpr (R < 32) {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_base(hcb_rx));
        for (uint32_t i = 0; i < kRx * kBf16 / 4; i++) {
            z[i] = 0;
        }
    }
    // NoC 1 multicasts run high -> low on Blackhole: the rectangle's end corner goes first (slice_mc.cpp)
    auto mc = [&](uint32_t b, uint32_t addr) {
        return get_noc_multicast_addr(get_common_arg_val<uint32_t>(b + 2), get_common_arg_val<uint32_t>(b + 3),
                                      get_common_arg_val<uint32_t>(b), get_common_arg_val<uint32_t>(b + 1), addr);
    };
    for (uint32_t l = 0; l < layers; l++) {
        const bool full = is_full(l);
        const uint32_t Dt = full ? kDtf : kDts;
        const bool kw = full ? h == 0 : h % 2 == 0;  // (the k norm weight: K heads)
        cb_reserve_back(hcb_w, 32);
        const uint32_t w = get_write_ptr(hcb_w);
        for (uint32_t d = 0; d < Dt; d++) {
            noc_async_read_page((2 * l) * 16 + d, wgen, w + d * kBf16);
            if (kw) {
                noc_async_read_page((2 * l + 1) * 16 + d, wgen, w + (16 + d) * kBf16);
            }
        }
        noc_async_read_barrier();
        cb_push_back(hcb_w, 32);
        cb_reserve_back(hcb_rx, kRx);
        noc_semaphore_wait_min(rx_sem, num_streamers * (l + 1));
        K2A_MARK(l, 0, kTlHead);
        if constexpr (R < 32) {  // (D16) the layer's q (and this head's k / v) tiles: rows 0 .. R - 1 of faces 0 / 1
            const uint32_t rx = get_write_ptr(hcb_rx), rx8 = cb_base(hcb_rx8);
            const bool kv = full ? h < 2 : true;
            for (uint32_t i = 0; i < kRx; i++) {
                if (i >= Dt && (i < kRxKV || !kv || i >= kRxKV + Dt)) {
                    continue;
                }
                for (uint32_t f = 0; f < 2; f++) {
                    local(rx8 + i * kTinyBytes + f * 32 * R, rx + i * kBf16 + f * 512, 32 * R);
                }
            }
            noc_async_read_barrier();
            K2A_MARK(l, 4, kTlHead);
        }
        cb_push_back(hcb_rx, kRx);
        // O: every live user's rows of this q head, then the head's Dt tiles into every streamer's o_in
        if (l == 0) {
            while (*hsched == 0) {
                invalidate_l1_cache();
            }
            if (*hsched != kSchedOk) {
                live = 0;
            }
        }
        noc_semaphore_wait_min(og_sem, live * (l + 1));
        K2A_MARK(l, 2, kTlHead);
        uint32_t o_in = *oaddr;
        while (o_in == 0) {
            invalidate_l1_cache();
            o_in = *oaddr;
        }
        uint32_t osrc = og;
        if constexpr (R < 32) {  // (D16) og's 32-row tiles -> 8-row tiles (rows 0 .. R - 1 of faces 0 / 1)
            osrc = cb_base(hcb_o8);
            for (uint32_t d = 0; d < Dt; d++) {
                for (uint32_t f = 0; f < 2; f++) {
                    local(og + d * kBf16 + f * 512, osrc + d * kTinyBytes + f * 32 * R, 32 * R);
                }
            }
            noc_async_read_barrier();
            K2A_MARK(l, 5, kTlHead);
        }
        for (uint32_t q = 0; q < nrect; q++) {
            const uint32_t b = rects + 1 + 5 * q;
            const uint64_t dst = mc(b, o_in + h * Dt * kTinyBytes);
            noc_async_write_multicast(osrc, dst, Dt * kTinyBytes, get_common_arg_val<uint32_t>(b + 4));
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < nrect; q++) {
            const uint32_t b = rects + 1 + 5 * q;
            const uint64_t dst = mc(b, get_semaphore(sem_o_id));
            noc_semaphore_inc_multicast(dst, 1, get_common_arg_val<uint32_t>(b + 4));
        }
        noc_async_atomic_barrier();
        K2A_MARK(l, 3, kTlHead);
    }
}
