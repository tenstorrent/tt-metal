// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused MoE expert step, reader (NCRISC, NOC0), one of the 72 worker cores.
// 1. Reads the top-8 expert ids and routing weights (row 0 of their TILE tensors) and the token row x
//    (row 0 of each of the 72 x tiles) into 1x32 "tiny" tiles.
// 2. Gate cores build the routing-weight tile (face 0 of a 32x32 tile = 8 tiny tiles, w_e repeated 32 times).
// 3. Streams this core's weight tiles, in the order compute eats them:
//      gate_up: for e in 0..7: for kt in 0..71: W_gu[id_e][kt][col_gu]
//      down:    for kt in 0..35: for e in 0..7: W_dn[id_e][kt][col_dn]   (matches the gathered hidden [kt][e])
//    in blocks of SBK tiles, nbuf - 1 blocks in flight (one transaction id per block).
// Runtime args: x, routing weights, ids, w_gate_up, w_down addresses; is_gate; col_gu; col_dn.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = get_named_compile_time_arg_val("cb_x");
constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
constexpr uint32_t cb_wd = get_named_compile_time_arg_val("cb_wd");
constexpr uint32_t cb_ws16 = get_named_compile_time_arg_val("cb_ws16");
constexpr uint32_t cb_scr = get_named_compile_time_arg_val("cb_scr");
constexpr uint32_t sbk = get_named_compile_time_arg_val("sbk");
constexpr uint32_t nbuf = get_named_compile_time_arg_val("nbuf");
constexpr uint32_t kt_gu = get_named_compile_time_arg_val("kt_gu");      // 72
constexpr uint32_t nt_gu = get_named_compile_time_arg_val("nt_gu");      // 72
constexpr uint32_t kt_dn = get_named_compile_time_arg_val("kt_dn");      // 36
constexpr uint32_t nt_dn = get_named_compile_time_arg_val("nt_dn");      // 72
constexpr uint32_t topk = get_named_compile_time_arg_val("topk");        // 8
constexpr uint32_t x_tiles = get_named_compile_time_arg_val("x_tiles");  // 72

constexpr uint32_t TINY = 64;         // bytes of a 1x32 bf16 tile: [face0 row0 | face1 row0]
constexpr uint32_t FACE_BYTES = 512;  // one 16x16 bf16 face of a 32x32 tile

struct Stream {
    uint32_t cb, base, end, wr, tile_bytes;
    uint32_t in_flight = 0, trid_issue = 1, trid_wait = 1;

    template <typename Acc>
    FORCE_INLINE void block(const Acc& acc, const uint32_t* pages) {
        // room for this block plus the ones still in flight
        cb_reserve_back(cb, sbk * (in_flight + 1));
        noc_async_read_set_trid(trid_issue);
        for (uint32_t i = 0; i < sbk; ++i) {
            noc_async_read(acc.get_noc_addr(pages[i]), wr, tile_bytes);
            wr += tile_bytes;
        }
        if (wr >= end) {
            wr = base;
        }
        trid_issue = trid_issue == nbuf ? 1 : trid_issue + 1;
        if (++in_flight == nbuf - 1) {
            retire();
        }
    }
    // one contiguous run of sbk tiles (stick layout: a column of one expert sits in one DRAM bank, in order)
    FORCE_INLINE void block_contig(uint64_t src) {
        cb_reserve_back(cb, sbk * (in_flight + 1));
        noc_async_read_set_trid(trid_issue);
        noc_async_read(src, wr, sbk * tile_bytes);
        wr += sbk * tile_bytes;
        if (wr >= end) {
            wr = base;
        }
        trid_issue = trid_issue == nbuf ? 1 : trid_issue + 1;
        if (++in_flight == nbuf - 1) {
            retire();
        }
    }
    FORCE_INLINE void retire() {
        noc_async_read_barrier_with_trid(trid_wait);
        cb_push_back(cb, sbk);
        trid_wait = trid_wait == nbuf ? 1 : trid_wait + 1;
        --in_flight;
    }
};

void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t wts_addr = get_arg_val<uint32_t>(a++);
    const uint32_t ids_addr = get_arg_val<uint32_t>(a++);
    const uint32_t gu_addr = get_arg_val<uint32_t>(a++);
    const uint32_t dn_addr = get_arg_val<uint32_t>(a++);
    const uint32_t is_gate = get_arg_val<uint32_t>(a++);
    const uint32_t col_gu = get_arg_val<uint32_t>(a++);
    const uint32_t col_dn = get_arg_val<uint32_t>(a++);

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto wts_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<wts_args.next_compile_time_args_offset()>();
    constexpr auto gu_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto dn_args = TensorAccessorArgs<gu_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto wts = TensorAccessor(wts_args, wts_addr);
    const auto ids = TensorAccessor(ids_args, ids_addr);
    const auto gu = TensorAccessor(gu_args, gu_addr);
    const auto dn = TensorAccessor(dn_args, dn_addr);

    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    // ids (uint16) and routing weights (bf16): row 0 of face 0 holds elements 0..15.
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(ids.get_noc_addr(0), scr, 32);
    noc_async_read(wts.get_noc_addr(0), scr + 32, 32);
    // x row 0 -> tiny tiles
    cb_reserve_back(cb_x, x_tiles);
    const uint32_t xl = get_write_ptr(cb_x);
    for (uint32_t t = 0; t < x_tiles; ++t) {
        noc_async_read(xs.get_noc_addr(t, 0), xl + t * TINY, 32);
        noc_async_read(xs.get_noc_addr(t, FACE_BYTES), xl + t * TINY + 32, 32);
    }
    noc_async_read_barrier();
    cb_push_back(cb_x, x_tiles);

    uint32_t eid[topk];
    volatile tt_l1_ptr uint16_t* id16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    for (uint32_t e = 0; e < topk; ++e) {
        eid[e] = id16[e];
    }
    if (is_gate) {
        volatile tt_l1_ptr uint16_t* w16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr + 32);
        cb_reserve_back(cb_ws16, 1);
        volatile tt_l1_ptr uint32_t* ws = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ws16));
        for (uint32_t e = 0; e < topk; ++e) {
            const uint32_t v = w16[e];
            const uint32_t pair = v | (v << 16);
            for (uint32_t i = 0; i < 16; ++i) {
                ws[e * 16 + i] = pair;
            }
        }
        cb_push_back(cb_ws16, 1);
    }

    // gate_up streams through cb_w, down through cb_wd (own format); trids 1..nbuf are reused, so the gate_up stream
    // is fully retired before the down stream starts.
    auto open = [](Stream& st, uint32_t cb) {
        st.cb = cb;
        st.tile_bytes = get_tile_size(cb);
        st.base = get_write_ptr(cb);
        st.end = st.base + nbuf * sbk * st.tile_bytes;
        st.wr = st.base;
    };
    Stream s, sd;
    open(s, cb_w);
    open(sd, cb_wd);
#ifdef STICK_LAYOUT
    // Stick layout (moe_expert.stick_weights): stick q = expert * n_cols + col is kt_* tiles at pages
    // n_banks * (m0 + i) + q % n_banks, m0 = (q / n_banks) * kt_*: contiguous in bank q % n_banks.
    constexpr uint32_t nb = get_named_compile_time_arg_val("n_banks");
    for (uint32_t e = 0; e < topk; ++e) {
        const uint32_t q = eid[e] * nt_gu + col_gu;
        const uint64_t src = gu.get_noc_addr((q / nb) * kt_gu * nb + q % nb);
        for (uint32_t kt = 0; kt < kt_gu; kt += sbk) {
            s.block_contig(src + kt * s.tile_bytes);
        }
    }
    while (s.in_flight) {
        s.retire();
    }
    // down: [e][kt] order (the gathered hidden is [e][kt] in this layout)
    for (uint32_t e = 0; e < topk; ++e) {
        const uint32_t q = eid[e] * nt_dn + col_dn;
        const uint64_t src = dn.get_noc_addr((q / nb) * kt_dn * nb + q % nb);
        for (uint32_t kt = 0; kt < kt_dn; kt += sbk) {
            sd.block_contig(src + kt * sd.tile_bytes);
        }
    }
#else
    uint32_t pages[sbk];

    // gate_up: one 72-tile column per expert
    for (uint32_t e = 0; e < topk; ++e) {
        const uint32_t first = eid[e] * kt_gu * nt_gu + col_gu;
        for (uint32_t kt = 0; kt < kt_gu; kt += sbk) {
            for (uint32_t i = 0; i < sbk; ++i) {
                pages[i] = first + (kt + i) * nt_gu;
            }
            s.block(gu, pages);
        }
    }
    while (s.in_flight) {
        s.retire();
    }
    // down: [kt][e] order
    uint32_t dfirst[topk];
    for (uint32_t e = 0; e < topk; ++e) {
        dfirst[e] = eid[e] * kt_dn * nt_dn + col_dn;
    }
    constexpr uint32_t dn_tiles = kt_dn * topk;
    for (uint32_t t = 0; t < dn_tiles; t += sbk) {
        for (uint32_t i = 0; i < sbk; ++i) {
            const uint32_t q = t + i;
            pages[i] = dfirst[q % topk] + (q / topk) * nt_dn;
        }
        sd.block(dn, pages);
    }
#endif
    while (sd.in_flight) {
        sd.retire();
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}
