// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert end pass, reader: for each token t of this core's range, its S rows (pages t S + k of the row-major
// result buffer, H bf16 a page: the K routed expert rows, then the shared expert's row) into CB 0, row k at tile slots
// 4 k .. 4 k + 3 (a row of 3840 elements read contiguously into 4 1024-element slots); and its K gate weights (fp32,
// page t of w) as bf16 at element 0 of K scalar tiles of CB 1, then 1.0 for the shared row.
// Up to NBUF tokens are in flight, token i's reads on transaction id 1 + i % NBUF into ring slot i % NBUF of CB 0 (the
// CB holds exactly NBUF tokens) and of the w scratch; they retire in order.
// Zeroed once: a row fills 3.75 slots, so the last 512 B of each row's 4th slot (always the same bytes: the ring holds
// whole tokens) stay zero (finite), as do the scalar tiles' other elements.
// Stamps (RT 4): 3 cycles waiting on the reads, 4 waiting for CB 0 space.
// RT: 0 rows addr, 1 w addr, 2 first token, 3 tokens, 4 stamps. CT: 0 K, 1 S, 2 row bytes, 3 w scratch L1 (64 B
// aligned, NBUF pages of 64 B), 4 NBUF, 5 CB 1 bytes, then TensorAccessorArgs(rows), (w).
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

FORCE_INLINE uint16_t bf16_rne(uint32_t f) { return static_cast<uint16_t>((f + 0x7FFF + ((f >> 16) & 1)) >> 16); }

FORCE_INLINE void zero_l1(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t w_scr = get_compile_time_arg_val(3);
    constexpr uint32_t NBUF = get_compile_time_arg_val(4);
    constexpr uint32_t w_cb_bytes = get_compile_time_arg_val(5);
    constexpr auto rargs = TensorAccessorArgs<6>();
    constexpr auto wargs = TensorAccessorArgs<rargs.next_compile_time_args_offset()>();
    constexpr uint32_t rows_cb = tt::CBIndex::c_0, w_cb = tt::CBIndex::c_1;
    constexpr uint32_t TPR = 4, tile_bytes = 2048, tok_pages = S * TPR, tok_bytes = tok_pages * tile_bytes;
    static_assert(row_bytes <= TPR * tile_bytes && NBUF >= 1 && NBUF < 16);

    const uint32_t t0 = get_arg_val<uint32_t>(2), nt = get_arg_val<uint32_t>(3), end = t0 + nt;
    const auto rs = TensorAccessor(rargs, get_arg_val<uint32_t>(0), row_bytes);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(1), 64);

    const uint32_t rows_base = get_write_ptr(rows_cb);
    for (uint32_t r = 0; r < NBUF * S; ++r) {
        zero_l1(rows_base + r * TPR * tile_bytes + row_bytes, TPR * tile_bytes - row_bytes);
    }
    zero_l1(get_write_ptr(w_cb), w_cb_bytes);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(4));
    uint32_t t_bar = 0, t_res = 0;
    uint32_t issued = t0, landed = t0;
    while (landed < end) {
        while (issued < end && issued - landed < NBUF) {
            const uint32_t a = wall();
            cb_reserve_back(rows_cb, (issued - landed + 1) * tok_pages);
            t_res += wall() - a;
            const uint32_t slot = (issued - t0) % NBUF, l1 = rows_base + slot * tok_bytes;
            noc_async_read_set_trid(1 + slot);
            for (uint32_t k = 0; k < S; ++k) {
                noc_async_read_page(issued * S + k, rs, l1 + k * TPR * tile_bytes);
            }
            noc_async_read_page(issued, ws, w_scr + slot * 64);
            noc_async_read_set_trid(0);
            ++issued;
        }
        const uint32_t slot = (landed - t0) % NBUF;
        const uint32_t a = wall();
        noc_async_read_barrier_with_trid(1 + slot);
        t_bar += wall() - a;
        cb_reserve_back(w_cb, S);
        const uint32_t wl = get_write_ptr(w_cb);
        volatile tt_l1_ptr uint32_t* wv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_scr + slot * 64);
        for (uint32_t k = 0; k < K; ++k) {
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wl + k * tile_bytes) = bf16_rne(wv[k]);
        }
        *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wl + K * tile_bytes) = 0x3F80;
        cb_push_back(w_cb, S);
        cb_push_back(rows_cb, tok_pages);
        ++landed;
    }
    stamps[3] = t_bar;
    stamps[4] = t_res;
}
