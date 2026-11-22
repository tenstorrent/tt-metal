// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter slice (BRISC; M6b), timeline variant (M6b-2a attribution; timeline_m6b.py selects it at
// runtime, d_slice_dm.cpp stays the verified default): d_slice_dm.cpp plus wall-clock marks into an L1 word array (one
// more runtime arg after the slices' (x, y) list). Per sublayer j, words 6 j + 0..4: the norm's exchange done, h
// multicast done, every streamer's coefficients in, conv_in multicast done, the all-reduce round's slots in; then
// 12 layers + 0 / 1: the final exchange, the final h multicast.
//
// d_slice_dm.cpp: one of the K cores per chip that own the drafter's residual stream by
// hidden columns (slice k: tiles [k T, (k + 1) T) of the R = 32 rows, as V 32 x 32 views), see d_slice_compute.cpp.
// slice_dm.cpp (the target's) with the drafter's per-sublayer flow. Per sublayer j = 2 l + s (s 0: attention, 1: MLP):
//   the sublayer's norm weight and conv bases (DRAM rows) for the compute; the exchange of the per-row partial sums
//   of squares (as the target's); the multicast of h into every streamer (their sem_slots + 1); once every streamer of
//   this chip has written its coefficient tiles into this slice's coefficient CB (sem_coef), those to the compute;
//   the multicast of conv_in(h) (xin / m_in); then the all-reduce round j's slots of every chip to the compute (the
//   streamers' partials and the peer hubs' fabric packets, as the target's).
// then the final norm and its h for the LM head. The bases share their CB with the expanded coefficients (the compute
// expands in place), so the next sublayer's bases are read only after its exchange: by then the compute has consumed
// the previous coefficients (its partial sum of squares follows their last use).
// Dumps (DRAM rows of K x V views): per layer l rows 6 l .. 6 l + 5: h, xin, x after attention, h2, m_in, x after the
// MLP; row 6 layers: the final h (always written; the rest with dump).
//
// Compile-time args: 0 num_chips, 1 chip, 2 k, 3 K, 4 T, 5 V, 6 layers, 7 local streamers (writing this slice's
// slots), 8 P (fabric packets per chip slot), 9 sem_slots (the streamers'), 10 sem_exch, 11 dump, 12 inject (the
// attention sublayers' y from DRAM rows: teacher forcing), 13 sem_coef, 14 num_streamers, 15 Mb (rows per block: the
// conv's shift stays inside a block), 16 coefficient tiles (48)
// Runtime args: 0 sem_haddr id (the streamers' cb_h address, from streamer 0), 1 / 2 slot semaphores (parity 0 / 1),
// 3 launch counter word, 4 x0 (DRAM rows of K x V views, row 0), 5 norm weights (rows: 2 l input norm, 2 l + 1 post
// norm, 2 layers the final norm), 6 conv bases (rows 4 j + section, row-replicated), 7 inject (rows: layer l's attention
// output), 8 dump, 9 sem_slotaddr id (the hub's and the streamers' mailbox for the slots address), 10 sem_coefaddr id
// (the streamers' mailbox for the coefficient CB address), 11 hub x | y << 16, 12 rectangles of the streamers, then per
// rectangle x0, y0, x1, y1, cores, then K x (x, y) of the slices.

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
constexpr uint32_t sem_slots = get_compile_time_arg_val(9);
constexpr uint32_t sem_exch = get_compile_time_arg_val(10);
constexpr bool dump = get_compile_time_arg_val(11) != 0;
constexpr bool inject = get_compile_time_arg_val(12) != 0;
constexpr uint32_t sem_coef = get_compile_time_arg_val(13);
constexpr uint32_t num_streamers = get_compile_time_arg_val(14);
constexpr uint32_t Mb = get_compile_time_arg_val(15);
constexpr uint32_t kCoefTiles = get_compile_time_arg_val(16);
static_assert(K > 1, "the norms' partial sums go through the exchange");

constexpr uint32_t cb_slots = 0, cb_x = 2, cb_hout = 4, cb_x0 = 5, cb_const = 6, cb_pss = 7, cb_exch = 8;
constexpr uint32_t cb_wpre = 11, cb_coef = 13, cb_base = 14, cb_inj = 16;
constexpr uint32_t kConstTiles = 20;  // eye, ones, fold, S, E_0 .. E_15 (d_slice_compute.cpp)
constexpr uint32_t kView = 2048, kF32Tile = 4096;
constexpr uint32_t slice_bytes = T * kView;
constexpr uint32_t slot_bytes = V * kView;
constexpr uint32_t rounds = 2 * layers;
constexpr uint32_t per_round = n_loc + (num_chips - 1) * P;

void kernel_main() {
    size_t a = 0;
    volatile tt_l1_ptr uint32_t* haddr_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(a++)));
    const uint32_t sem_addr[2] = {get_arg_val<uint32_t>(a), get_arg_val<uint32_t>(a + 1)};
    a += 2;
    const uint32_t launch_addr = get_arg_val<uint32_t>(a++);
    const uint32_t x0_addr = get_arg_val<uint32_t>(a++);
    const uint32_t wnorm_addr = get_arg_val<uint32_t>(a++);
    const uint32_t base_addr = get_arg_val<uint32_t>(a++);
    const uint32_t inject_addr = get_arg_val<uint32_t>(a++);
    const uint32_t dump_addr = get_arg_val<uint32_t>(a++);
    const uint32_t slotaddr_box = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t coefaddr_box = get_semaphore(get_arg_val<uint32_t>(a++));
    const uint32_t hub_xy = get_arg_val<uint32_t>(a++);
    const uint32_t rects = get_arg_val<uint32_t>(a++);
    const size_t rect_args = a;
    a += 5 * rects;
    const size_t slice_args = a;
    volatile tt_l1_ptr uint32_t* tl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(a + 2 * K));
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t w) { tl[w] = *clk; };
    auto rect = [&](uint32_t q, uint32_t i) { return get_arg_val<uint32_t>(rect_args + 5 * q + i); };
    auto cb_addr = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    const uint32_t slots = cb_addr(cb_slots), exch_addr = cb_addr(cb_exch);
    if constexpr (k == 0) {
        // the slots address to the hub and the streamers, the coefficient CB's to the streamers (two words of the
        // const CB, 16 B aligned as the mailboxes, overwritten by the tiles below once sent)
        const uint32_t word = get_write_ptr(cb_const);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word) = slots;
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(word + 16) = cb_addr(cb_coef);
        noc_semaphore_set_remote(word, get_noc_addr(hub_xy & 0xffff, hub_xy >> 16, slotaddr_box));
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_set_multicast(
                word, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), slotaddr_box), rect(q, 4));
            noc_semaphore_set_multicast(
                word + 16, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), coefaddr_box),
                rect(q, 4));
        }
        noc_async_write_barrier();
    }

    // constant tiles (bf16 32x32, row-major element (row, col) at face 2 (row / 16) + col / 16): eye, ones, fold (R =
    // 32: the eye), S (the in-block shift: S[r][r - 1] = 1 unless r starts a block), E_m (coefficient columns 2 m, 2 m +
    // 1 onto channels 0 .. 15, 16 .. 31)
    cb_reserve_back(cb_const, kConstTiles);
    {
        volatile tt_l1_ptr uint16_t* c = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_const));
        constexpr uint16_t one = 0x3f80;
        for (uint32_t i = 0; i < 1024; i++) {
            const uint32_t face = i / 256, row = (face / 2) * 16 + (i % 256) / 16, col = (face % 2) * 16 + i % 16;
            c[i] = row == col ? one : 0;
            c[1024 + i] = one;
            c[2048 + i] = row == col ? one : 0;
            c[3072 + i] = (row % Mb != 0 && col + 1 == row) ? one : 0;
            for (uint32_t m = 0; m < 16; m++) {
                c[(4 + m) * 1024 + i] = (row == 2 * m + (col >= 16 ? 1 : 0)) ? one : 0;
            }
        }
    }
    cb_push_back(cb_const, kConstTiles);

    // the views' tails (past the slice's real bytes) of every slot are zero: nothing else writes them
    if constexpr (slot_bytes > slice_bytes) {
        for (uint32_t s = 0; s < 2 * num_chips; s++) {
            volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots + s * slot_bytes);
            for (uint32_t i = slice_bytes / 4; i < slot_bytes / 4; i++) {
                t[i] = 0;
            }
        }
    }

    const InterleavedAddrGen<true> x0g{.bank_base_address = x0_addr, .page_size = K * slot_bytes};
    const InterleavedAddrGen<true> wnorm{.bank_base_address = wnorm_addr, .page_size = K * slot_bytes};
    const InterleavedAddrGen<true> baseg{.bank_base_address = base_addr, .page_size = K * slot_bytes};
    const InterleavedAddrGen<true> injg{.bank_base_address = inject_addr, .page_size = K * slot_bytes};
    const InterleavedAddrGen<true> dumpg{.bank_base_address = dump_addr, .page_size = K * slot_bytes};

    // seed: this slice's views of x0
    cb_reserve_back(cb_x0, V);
    noc_async_read(x0g.get_noc_addr(0, k * slot_bytes), get_write_ptr(cb_x0), slot_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_x0, V);

    volatile tt_l1_ptr uint32_t* launch_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(launch_addr);
    invalidate_l1_cache();
    const uint32_t launch = *launch_word;
    uint32_t expected[2] = {launch * ((rounds + 1) / 2) * per_round, launch * (rounds / 2) * per_round};
    volatile tt_l1_ptr uint32_t* slot_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr[0]),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr[1])};
    volatile tt_l1_ptr uint32_t* exch_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_exch));
    const uint32_t exch_sem_addr = get_semaphore(sem_exch);
    volatile tt_l1_ptr uint32_t* coef_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_coef));
    const uint32_t x_addr = cb_addr(cb_x);  // x after sublayer j sits at view ((j + 1) & 1) V of cb_x

    uint32_t e = 0;  // exchanges so far
    auto exchange = [&]() {
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
        e++;
    };
    // the compute's cb_hout views into every streamer's cb_h, then bump their sem_slots; dump row `row` (or none)
    uint32_t h_dst = 0;
    auto mcast = [&](bool dump_it, uint32_t row) {
        while (h_dst == 0) {
            invalidate_l1_cache();
            h_dst = *haddr_box;
        }
        cb_wait_front(cb_hout, V);
        const uint32_t src = get_read_ptr(cb_hout);
        const uint32_t dst = h_dst + k * slice_bytes;
        for (uint32_t q = 0; q < rects; q++) {
            noc_async_write_multicast(
                src, get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), dst), slice_bytes, rect(q, 4));
        }
        noc_async_write_barrier();
        for (uint32_t q = 0; q < rects; q++) {
            noc_semaphore_inc_multicast(
                get_noc_multicast_addr(rect(q, 0), rect(q, 1), rect(q, 2), rect(q, 3), get_semaphore(sem_slots)),
                1,
                rect(q, 4));
        }
        if (dump_it) {
            noc_async_write(src, dumpg.get_noc_addr(row, k * slot_bytes), slot_bytes);
        }
        noc_async_write_barrier();
        noc_async_atomic_barrier();
        cb_pop_front(cb_hout, V);
    };
    auto dump_x = [&](uint32_t j, uint32_t row) {
        if constexpr (dump) {
            noc_async_write(x_addr + ((j + 1) & 1) * slot_bytes, dumpg.get_noc_addr(row, k * slot_bytes), slot_bytes);
        }
    };

    for (uint32_t l = 0; l < layers; l++) {
        if constexpr (inject) {
            cb_reserve_back(cb_inj, V);
            noc_async_read(injg.get_noc_addr(l, k * slot_bytes), get_write_ptr(cb_inj), slot_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_inj, V);
        }
        for (uint32_t s = 0; s < 2; s++) {
            const uint32_t j = 2 * l + s, par = j & 1;
            cb_reserve_back(cb_wpre, V);
            noc_async_read(wnorm.get_noc_addr(j, k * slot_bytes), get_write_ptr(cb_wpre), slot_bytes);
            noc_async_read_barrier();
            cb_push_back(cb_wpre, V);
            exchange();  // the sublayer's norm
            mark(6 * j + 0);
            if (j > 0) {
                dump_x(j - 1, s == 0 ? 6 * l - 1 : 6 * l + 2);  // x after the previous sublayer
            }
            // the bases (their CB is the coefficients'): consumed coefficients by now, see above
            cb_reserve_back(cb_base, 4 * V);
            for (uint32_t q = 0; q < 4; q++) {
                noc_async_read(baseg.get_noc_addr(4 * j + q, k * slot_bytes), get_write_ptr(cb_base) + q * slot_bytes, slot_bytes);
            }
            mcast(dump, 6 * l + 3 * s);  // h / h2
            mark(6 * j + 1);
            noc_async_read_barrier();
            cb_push_back(cb_base, 4 * V);
            // every streamer of this chip has written its coefficient tiles of this sublayer
            noc_semaphore_wait_min(coef_sem, num_streamers * (j + 1));
            mark(6 * j + 2);
            cb_reserve_back(cb_coef, kCoefTiles);
            cb_push_back(cb_coef, kCoefTiles);
            mcast(dump, 6 * l + 3 * s + 1);  // xin / m_in
            mark(6 * j + 3);
            // the all-reduce round j
            expected[par] += per_round;
            noc_semaphore_wait_min(slot_sem[par], expected[par]);
            mark(6 * j + 4);
            cb_reserve_back(cb_slots, num_chips * V);
            cb_push_back(cb_slots, num_chips * V);
        }
    }
    // the final norm: h for the LM head
    cb_reserve_back(cb_wpre, V);
    noc_async_read(wnorm.get_noc_addr(rounds, k * slot_bytes), get_write_ptr(cb_wpre), slot_bytes);
    noc_async_read_barrier();
    cb_push_back(cb_wpre, V);
    exchange();
    mark(12 * layers + 0);
    dump_x(rounds - 1, 6 * layers - 1);
    mcast(true, 6 * layers);
    mark(12 * layers + 1);
    *launch_word = launch + 1;
}
