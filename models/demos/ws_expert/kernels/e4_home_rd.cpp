// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// EP-4 home pass (ep4_layer.build_home), reader (BRISC, NOC0: the DRAM reads). Gate: core 0 holds the counters the
// row's ports increment (word c: partials landed from the row's chip c); it reads the expected counts (the lists' ready
// counts page, words EPC + 1 + c), waits until every other chip's counter reaches its count, clears them for the next
// layer and raises semaphore 0 on every core; every core waits for it before reading P.
// Per own token t of the core's range: its topk row (the ids: the row's chips c holding >= 1 of its experts, the mask)
// and P pages t S5 + c for the slots c < C in the mask into CB 0 slot c (4 tiles; an absent slot's row is zeroed in L1
// instead: P is never read where nobody wrote), then the shared expert's slot (C) always; CB 1: S5 scalar tiles of 1.0
// (written once: a constant). The compute (we_compute.cpp, S5 rows) sums them in slot order.
// CT: 0 S5, 1 C, 2 row bytes, 3 EPC, 4 K, 5 NBUF, 6 tk table (L1, 64 B aligned, 128 B a token of the range), 7 counters
//     (L1, core 0), 8 counts page scratch (L1, 64 B aligned), 9 page bytes of the ready counts, then
//     TensorAccessorArgs(P), (tk), (rcounts).
// RT: 0 P addr, 1 tk addr, 2 rcounts addr, 3 t0, 4 nt, 5 CS, 6 me, 7 NC, 8 stamps (3 read waits, 4 CB space waits, 9
// gate
//     wait cycles), 9 + k core k's NoC x | y << 8.
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }
FORCE_INLINE void zero_l1(uint32_t addr, uint32_t bytes) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; ++i) {
        p[i] = 0;
    }
}

void kernel_main() {
    constexpr uint32_t S5 = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t EPC = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t NBUF = get_compile_time_arg_val(5);
    constexpr uint32_t tks = get_compile_time_arg_val(6);
    constexpr uint32_t cnt = get_compile_time_arg_val(7);
    constexpr uint32_t rcs = get_compile_time_arg_val(8);
    constexpr uint32_t rc_bytes = get_compile_time_arg_val(9);
    constexpr auto pargs = TensorAccessorArgs<10>();
    constexpr auto targs = TensorAccessorArgs<pargs.next_compile_time_args_offset()>();
    constexpr auto qargs = TensorAccessorArgs<targs.next_compile_time_args_offset()>();
    constexpr uint32_t rows_cb = tt::CBIndex::c_0, w_cb = tt::CBIndex::c_1;
    constexpr uint32_t TPR = 4, tile_bytes = 2048, tok_pages = S5 * TPR, tok_bytes = tok_pages * tile_bytes;
    static_assert(S5 == C + 1 && NBUF >= 1 && NBUF < 16);

    const auto ps = TensorAccessor(pargs, get_arg_val<uint32_t>(0), row_bytes);
    const auto ts = TensorAccessor(targs, get_arg_val<uint32_t>(1), 4096);  // (topk rows: 32 tokens a page)
    const auto qs = TensorAccessor(qargs, get_arg_val<uint32_t>(2), rc_bytes);
    const uint32_t t0 = get_arg_val<uint32_t>(3), nt = get_arg_val<uint32_t>(4), end = t0 + nt;
    const uint32_t CS = get_arg_val<uint32_t>(5), me = get_arg_val<uint32_t>(6), NC = get_arg_val<uint32_t>(7);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(8));
    auto core_at = [&](uint32_t k, uint32_t addr) {
        const uint32_t v = get_arg_val<uint32_t>(9 + k);
        return get_noc_addr(v & 0xFF, v >> 8, addr);
    };
    auto w = [](uint32_t addr) -> volatile tt_l1_ptr uint32_t& {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    };
    const uint32_t rows_base = get_write_ptr(rows_cb);
    for (uint32_t r = 0; r < NBUF * S5; ++r) {
        zero_l1(rows_base + r * TPR * tile_bytes + row_bytes, TPR * tile_bytes - row_bytes);
    }
    {  // CB 1: every page's S5 scalar tiles 1.0 (bf16 at element 0, the rest 0)
        const uint32_t wb = get_write_ptr(w_cb);
        zero_l1(wb, 2 * S5 * tile_bytes);
        for (uint32_t k = 0; k < 2 * S5; ++k) {
            *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(wb + k * tile_bytes) = 0x3F80;
        }
    }
    for (uint32_t t = t0; t < end; ++t) {  // (the range's topk rows: read once, overlapping the gate's wait)
        noc_async_read(ts.get_noc_addr(t / 32, t % 32 * 128), tks + (t - t0) * 128, 128);
    }
    // ---- gate
    const uint32_t a0 = wall();
    if (me == 0) {
        noc_async_read(qs.get_noc_addr(0), rcs, rc_bytes);
        noc_async_read_barrier();
        for (uint32_t c = 0; c < C; ++c) {
            if (c == CS) {
                continue;
            }
            const uint32_t want = w(rcs + (EPC + 1 + c) * 4);
            while (w(cnt + c * 16) < want) {
                invalidate_l1_cache();
            }
            w(cnt + c * 16) = 0;
        }
        const uint32_t go = get_semaphore(0);
        for (uint32_t k = 0; k < NC; ++k) {
            noc_semaphore_inc(core_at(k, go), 1);
        }
        noc_async_atomic_barrier();
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 1);
    noc_async_read_barrier();
    stamps[9] = wall() - a0;

    uint32_t t_bar = 0, t_res = 0;
    uint32_t issued = t0, landed = t0;
    while (landed < end) {
        while (issued < end && issued - landed < NBUF) {
            const uint32_t s0 = wall();
            cb_reserve_back(rows_cb, (issued - landed + 1) * tok_pages);
            t_res += wall() - s0;
            const uint32_t slot = (issued - t0) % NBUF, l1 = rows_base + slot * tok_bytes;
            uint32_t mask = 0;
            for (uint32_t k = 0; k < K; ++k) {
                mask |= 1u << (w(tks + (issued - t0) * 128 + (16 + k) * 4) / EPC);
            }
            noc_async_read_set_trid(1 + slot);
            for (uint32_t c = 0; c < S5; ++c) {
                if (c == C || ((mask >> c) & 1)) {
                    noc_async_read_page(issued * S5 + c, ps, l1 + c * TPR * tile_bytes);
                } else {
                    zero_l1(l1 + c * TPR * tile_bytes, row_bytes);
                }
            }
            noc_async_read_set_trid(0);
            ++issued;
        }
        const uint32_t slot = (landed - t0) % NBUF;
        const uint32_t s0 = wall();
        noc_async_read_barrier_with_trid(1 + slot);
        t_bar += wall() - s0;
        cb_reserve_back(w_cb, S5);
        cb_push_back(w_cb, S5);
        cb_push_back(rows_cb, tok_pages);
        ++landed;
    }
    stamps[3] = t_bar;
    stamps[4] = t_res;
}
