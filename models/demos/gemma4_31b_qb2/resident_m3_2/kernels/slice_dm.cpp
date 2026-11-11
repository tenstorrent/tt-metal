// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident slice (BRISC): one of the K cores per chip that own the residual stream by hidden columns (slice k:
// tiles [k T, (k + 1) T) of every row, as V zero-padded 32x32 views). Per step:
//   seed: x0 (this slice's tiles of the token rows' scaled embeddings, DRAM) for the compute; the compute's partial
//         sum of squares goes through the exchange below; its h multicast into every streamer.
//   all-reduce round r = 1 .. rounds: this round's post-norm weight (DRAM) for the compute; wait until the slot of
//         every chip is in (this chip's streamers write theirs, the peer hubs' fabric packets the others', all counting
//         on the slot semaphore of the round's parity) and hand the slots to the compute; two exchanges (post norm,
//         pre norm); multicast the compute's h tiles into every streamer and bump their sem_slots.
// Exchange (K > 1): the compute's partial per-row sums of squares (one fp32 32x32 tile) into every slice's exchange
// buffer [parity][k], then once all K are in, the K tiles to the compute. New for this prototype; the slice
// split itself replaces the single hub compute of changh95/qwen3.8-27b-p300x2 (qwen36_resident, Apache-2.0).
// The slot semaphores are global and monotonic (a peer chip can send the next launch's round 0 first), based on a
// per-slice launch counter the host zeroes while idle.
//
// Compile-time args: 0 num_chips, 1 chip, 2 k, 3 K, 4 R, 5 T, 6 V, 7 rounds, 8 local streamers (writing this slice),
// 9 P (fabric packets per chip slot), 10 sem_slots (the streamers'), 11 sem_exch, 12 dump_rounds, 13 pre_weighted,
// 14 M
// Runtime args: 0 sem_haddr id (this core's mailbox for the streamers' h address, from streamer 0), 1 / 2 slot
// semaphores (parity 0 / 1), 3 launch counter word, 4 tokens (DRAM [1, 1, 1, 32] uint32 row: the launch's x0 is
// gathered from the scaled embedding rows of tokens 0 .. M - 1), 5 post-norm weights (DRAM rows of K x V views),
// 6 sem_slotaddr id (the hub's and the streamers' mailbox for the slots address, published by slice 0), 7 hub
// x | y << 16, 8 dump (DRAM rows of K x V views: [0] final h, [1 + r] h of round r with dump_rounds), 9 timeline (L1,
// 0 = off), 10 scaled embedding table (DRAM rows of HIDDEN bf16), 11 status (DRAM [1, 1, 1, 32] uint32 row: word 0,
// the launch counter, picks the taps buffer), 12 rectangles, then per rectangle x0, y0, x1, y1, cores, then K x (x, y)
// of the slices, then the taps: two DRAM TILE buffers [1, 1, 32, n x HIDDEN] bf16 (parity = launch counter & 1;
// 0 = none; tap i at column tiles [i Ht, (i + 1) Ht), rows 0 .. R - 1) and the tapped layers as a 64-bit mask (lo,
// hi): the residual x after each tapped layer (DFlash taps / per-layer checks), written once its round's h is out
// (x stays put until two rounds later, which need slots this core hands over only after the write); then
// (pre_weighted) the pre-norm weights: DRAM rows of K x V views, row r the weight of round r's h (the input norm
// before a mixer half, the pre-feedforward norm before an MLP half, the final norm before the LM head), as the ttnn
// path's rms_norm(x, w) ahead of unfolded weights.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t num_chips = get_compile_time_arg_val(0);
constexpr uint32_t chip = get_compile_time_arg_val(1);
constexpr uint32_t k = get_compile_time_arg_val(2);
constexpr uint32_t K = get_compile_time_arg_val(3);
constexpr uint32_t R = get_compile_time_arg_val(4);
constexpr uint32_t T = get_compile_time_arg_val(5);
constexpr uint32_t V = get_compile_time_arg_val(6);
constexpr uint32_t rounds = get_compile_time_arg_val(7);
constexpr uint32_t n_loc = get_compile_time_arg_val(8);
constexpr uint32_t P = get_compile_time_arg_val(9);
constexpr uint32_t sem_slots = get_compile_time_arg_val(10);
constexpr uint32_t sem_exch = get_compile_time_arg_val(11);
constexpr bool dump_rounds = get_compile_time_arg_val(12) != 0;
constexpr bool pre_weighted = get_compile_time_arg_val(13) != 0;  // raw weights: h = rmsnorm(x) * w_pre (see below)
constexpr uint32_t M = get_compile_time_arg_val(14);               // rows of the launch (x0 rows M .. R - 1 zero)
// (M2b F2) the h multicast's rectangles [0, mc_split) go out from here (NOC 0), the rest from slice_mc.cpp (NCRISC,
// NOC 1); 0: all from here
constexpr uint32_t mc_split = get_compile_time_arg_val(15);
constexpr uint32_t sem_mc_go = get_compile_time_arg_val(16), sem_mc_done = get_compile_time_arg_val(17);
constexpr uint32_t sem_mc_src = get_compile_time_arg_val(18);

constexpr uint32_t cb_slots = 0, cb_w = 1, cb_x = 2, cb_hout = 4, cb_x0 = 5, cb_const = 6, cb_pss = 7, cb_exch = 8;
constexpr uint32_t cb_wpre = 11;
// (the slots and the exchange buffer are CBs: every slice, and every chip's, has them at the same address)
constexpr uint32_t kView = 2048, kF32Tile = 4096;
constexpr uint32_t kVocab = 262144, kHiddenBytes = 5376 * 2, kHt = 168;
constexpr uint32_t kTileBytes = 64 * R;
constexpr uint32_t slice_bytes = T * kTileBytes;  // real bytes of this slice's tiles (the views' tail is zero)
constexpr uint32_t slot_bytes = V * kView;
constexpr uint32_t per_round = n_loc + (num_chips - 1) * P;  // slot semaphore counts per round
constexpr uint32_t kTs = 4;

void kernel_main() {
    size_t a = 0;
    volatile tt_l1_ptr uint32_t* haddr_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(a++)));
    const uint32_t sem_addr[2] = {get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1)};
    a += 2;
    const uint32_t launch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tokens_addr = get_arg_val<uint32_t>(a++);
    const uint32_t wpost_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slotaddr_box = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t hub_xy = get_arg_val<uint32_t>(a++);
    const uint32_t dump_addr = get_arg_val<uint32_t>(a++);
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a++));
    const uint32_t embed_addr = get_arg_val<uint32_t>(a++);
    const uint32_t status_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rects = get_arg_val<uint32_t>(a++);
    const size_t rect_args = a;
    a += 5 * rects;
    const size_t slice_args = a;
    a += 2 * K;
    const uint32_t taps_buf[2] = {get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1)};
    a += 2;
    const uint32_t tap_lo = get_arg_val<uint32_t>(a++), tap_hi = get_arg_val<uint32_t>(a++);
    auto tapped = [&](uint32_t l) { return ((l < 32 ? tap_lo >> l : tap_hi >> (l - 32)) & 1) != 0; };
    const uint32_t wpre_addr = get_arg_val<uint32_t>(a++);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t w) {
        if (ts != nullptr) {
            ts[w] = *clk;
        }
    };
    mark(0);
    auto rect = [&](uint32_t q, uint32_t i) { return get_arg_val<uint32_t>(rect_args + 5 * q + i); };
    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    const uint32_t slots = cb_base(cb_slots), exch_addr = cb_base(cb_exch);
    if constexpr (k == 0) {
        // the slots address to the hub and the streamers (the const CB's first word, 16 B aligned as the mailbox,
        // overwritten by the tiles below once sent)
        const uint32_t word = get_write_ptr(cb_const);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = slots;
        noc_semaphore_set_remote(word, get_noc_addr(hub_xy & 0xffff, hub_xy >> 16, slotaddr_box));
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_set_multicast(
                word, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), slotaddr_box), rect(q, 4));
        }
        noc_async_write_barrier();
    }

    // the norm's constant tiles (bf16 32x32): eye, ones, fold (1 where row = column mod R)
    cb_reserve_back(cb_const, 3);
    {
        volatile tt_l1_ptr uint16_t* c = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_const));
        for (uint32_t i = 0; i < 1024; i++) {
            const uint32_t face = i / 256, row = (face / 2) * 16 + (i % 256) / 16, col = (face % 2) * 16 + i % 16;
            c[i] = row == col ? 0x3f80 : 0;
            c[1024 + i] = 0x3f80;
            c[2048 + i] = row % R == col % R ? 0x3f80 : 0;
        }
    }
    cb_push_back(cb_const, 3);

    // the views' tails (past the slice's real bytes) of every slot are zero: nothing else writes them
    if constexpr (slot_bytes > slice_bytes) {
        for (uint32_t s = 0; s < 2 * num_chips; s++) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots + s * slot_bytes);
            for (uint32_t i = slice_bytes / 4; i < slot_bytes / 4; i++) {
                t[i] = 0;
            }
        }
    }

    // seed: this slice's tiles of x0, the scaled embedding rows of the launch's tokens (rows M .. R - 1 zero); the
    // tokens and the status row (the taps parity) land in cb_w first, unused until round 1
    uint32_t taps_addr = 0;
    {
        const uint32_t words = cb_base(cb_w);
        volatile tt_l1_ptr uint32_t* tk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words);
        const InterleavedAddrGen<true> tg{.bank_base_address = tokens_addr, .page_size = 128};
        const InterleavedAddrGen<true> sg{.bank_base_address = status_addr, .page_size = 128};
        noc_async_read(tg.get_noc_addr(0), words, 128);
        noc_async_read(sg.get_noc_addr(0), words + 128, 128);
        noc_async_read_barrier();
        taps_addr = taps_buf[tk[32] & 1];
        cb_reserve_back(cb_x0, V);
        const uint32_t dst = get_write_ptr(cb_x0);
        const InterleavedAddrGen<true> emb{.bank_base_address = embed_addr, .page_size = kHiddenBytes};
        if constexpr (R == 1) {
            const uint32_t t = tk[0] < kVocab ? tk[0] : 0;
            noc_async_read(emb.get_noc_addr(t, k * slice_bytes), dst, slice_bytes);
            noc_async_read_barrier();
        } else {
            // row r: the embedding row (64 B aligned DRAM reads), then its 32-column chunks into the R x 32 tiles
            // (face 0: columns 0 .. 15, face 1: 16 .. 31; row r of a face at r x 32 B)
            const uint32_t row = words + 256;  // cb_w has 2 V views of room: >= one row
            for (uint32_t r = 0; r < R; r++) {
                if (r < M) {
                    const uint32_t t = tk[r] < kVocab ? tk[r] : 0;
                    noc_async_read(emb.get_noc_addr(t, 0), row, kHiddenBytes);
                    noc_async_read_barrier();
                }
                for (uint32_t c = 0; c < T; c++) {
                    for (uint32_t f = 0; f < 2; f++) {
                        volatile tt_l1_ptr uint32_t* d =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + c * kTileBytes + f * 32 * R + r * 32);
                        volatile tt_l1_ptr uint32_t* src =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row + ((k * T + c) * 32 + f * 16) * 2);
                        for (uint32_t w = 0; w < 8; w++) {
                            d[w] = r < M ? src[w] : 0;
                        }
                    }
                }
            }
        }
        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
        for (uint32_t i = slice_bytes / 4; i < slot_bytes / 4; i++) {
            t[i] = 0;
        }
        cb_push_back(cb_x0, V);
    }

    volatile tt_l1_ptr uint32_t* launch_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(launch_addr);
    invalidate_l1_cache();
    const uint32_t launch = *launch_word;
    uint32_t expected[2] = {launch * ((rounds + 1) / 2) * per_round, launch * (rounds / 2) * per_round};
    volatile tt_l1_ptr uint32_t* slot_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr[0]),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr[1])};
    volatile tt_l1_ptr uint32_t* exch_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_exch));
    const uint32_t exch_sem_addr = get_semaphore(sem_exch);
    const InterleavedAddrGen<true> wpost{.bank_base_address = wpost_addr, .page_size = K * slot_bytes};
    const InterleavedAddrGen<true> wpre{.bank_base_address = wpre_addr, .page_size = K * slot_bytes};
    // round r's pre-norm weight for the compute (issued with the round's other reads, pushed after their barrier)
    auto read_wpre = [&](uint32_t r) {
        if constexpr (pre_weighted) {
            cb_reserve_back(cb_wpre, V);
            noc_async_read(wpre.get_noc_addr(r, k * slot_bytes), get_write_ptr(cb_wpre), slot_bytes);
        }
    };
    auto push_wpre = [&]() {
        if constexpr (pre_weighted) {
            cb_push_back(cb_wpre, V);
        }
    };
    const InterleavedAddrGen<true> dump{.bank_base_address = dump_addr, .page_size = K * slot_bytes};
    const uint32_t x_addr = cb_base(cb_x);  // x of an even round sits at the CB's base (see slice_compute.cpp)
    uint32_t tap_index = 0;

    uint32_t e = 0;  // exchanges so far
    auto exchange = [&]() {
        if constexpr (K > 1) {
            cb_wait_front(cb_pss, 1);
            const uint32_t src = get_read_ptr(cb_pss);
            const uint32_t dst = exch_addr + ((e & 1) * K + k) * kF32Tile;
            for (uint32_t j = 0; j < K; j++) {
                const uint32_t sx = get_arg_val<uint32_t>(slice_args + 2 * j);
                const uint32_t sy = get_arg_val<uint32_t>(slice_args + 2 * j + 1);
                noc_async_write(src, get_noc_addr(sx, sy, dst), kF32Tile);
            }
            noc_async_write_barrier();
            for (uint32_t j = 0; j < K; j++) {
                const uint32_t sx = get_arg_val<uint32_t>(slice_args + 2 * j);
                const uint32_t sy = get_arg_val<uint32_t>(slice_args + 2 * j + 1);
                noc_semaphore_inc(get_noc_addr(sx, sy, exch_sem_addr), 1);
            }
            cb_pop_front(cb_pss, 1);
            noc_semaphore_wait_min(exch_sem, K * (e + 1));
            cb_reserve_back(cb_exch, K);
            cb_push_back(cb_exch, K);
        }
        e++;
    };
    // the compute's h tiles of round r into every streamer, then bump their sem_slots
    uint32_t h_dst = 0;
    auto mcast_h = [&](uint32_t r) {
        while (h_dst == 0) {
            invalidate_l1_cache();
            h_dst = *haddr_box;
        }
        cb_wait_front(cb_hout, V);
        const uint32_t src = get_read_ptr(cb_hout);
        const uint32_t dst = h_dst + k * slice_bytes;
        const uint32_t my_rects = mc_split ? mc_split : rects;
        if constexpr (mc_split != 0) {
            // the NCRISC multicasts the other rectangles from the same front (cb_hout keeps this RISC as consumer)
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_mc_src)) = src;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_mc_go)) = r + 1;
        }
        for (uint32_t q = 0; q < my_rects; q++) {
            noc_async_write_multicast(
                src, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), dst), slice_bytes, rect(q, 4));
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < my_rects; q++) {
            noc_semaphore_inc_multicast(
                get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), get_semaphore(sem_slots)),
                1,
                rect(q, 4));
        }
        if (r == rounds) {
            noc_async_write(src, dump.get_noc_addr(0, k * slot_bytes), slot_bytes);
        }
        if constexpr (dump_rounds) {
            noc_async_write(src, dump.get_noc_addr(1 + r, k * slot_bytes), slot_bytes);
        }
        if (taps_addr && r >= 2 && r % 2 == 0 && tapped(r / 2 - 1)) {
            // this slice's R x 32 tiles of x into tile row 0 of the taps tensor (rows 0 .. R - 1 of faces 0 / 1)
            const InterleavedAddrGen<true> tg{.bank_base_address = taps_addr, .page_size = kView};
            const uint32_t base = tap_index++ * kHt + k * T;
            for (uint32_t c = 0; c < T; c++) {
                for (uint32_t f = 0; f < 2; f++) {
                    noc_async_write(x_addr + c * kTileBytes + f * 32 * R, tg.get_noc_addr(base + c, f * 512), 32 * R);
                }
            }
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        if constexpr (mc_split != 0) {
            volatile tt_l1_ptr uint32_t* mc_done =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_mc_done));
            do {
                invalidate_l1_cache();
            } while (*mc_done < r + 1);
        }
        cb_pop_front(cb_hout, V);
    };

    read_wpre(0);
    noc_async_read_barrier();
    push_wpre();
    exchange();  // the seed's pre norm
    mcast_h(0);
    for (uint32_t r = 1; r <= rounds; r++) {
        const uint32_t ar = r - 1, par = ar & 1;
        cb_reserve_back(cb_w, V);
        noc_async_read(wpost.get_noc_addr(ar, k * slot_bytes), get_write_ptr(cb_w), slot_bytes);
        read_wpre(r);
        expected[par] += per_round;
        noc_semaphore_wait_min(slot_sem[par], expected[par]);
        mark(2 + kTs * ar);
        noc_async_read_barrier();
        cb_push_back(cb_w, V);
        push_wpre();
        cb_reserve_back(cb_slots, num_chips * V);
        cb_push_back(cb_slots, num_chips * V);
        exchange();  // post norm
        mark(2 + kTs * ar + 1);
        exchange();  // pre norm
        mark(2 + kTs * ar + 2);
        mcast_h(r);
        mark(2 + kTs * ar + 3);
    }
    *launch_word = launch + 1;
    mark(1);
}
