// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 2 receiver (BRISC, NOC1) on a gate/up core. Loads this core's weights (resident in CB 1) once, then a
// non-blocking loop:
//   * x: publishes each block to compute (CB 0 = the x ring) once its XP pieces have arrived (XARR); counts the blocks
//     compute has freed and posts them (XCONS) for the forwarder,
//   * MODE 1, partials out: each sub-block's partials for the partner (CB 17, packed by compute) are written into the
//     partner's partial ring (its CB 2, same address) once the partner has a free slot (PFREE here), then the
//     partner's PARR is bumped,
//   * MODE 1, partials in: the partner's partials (PARR) are published to compute (CB 2); slots compute has freed are
//     credited back to the partner (its PFREE),
//   * h: drained (CB 16); the last sub-block's h tiles are written to DRAM (pages h_first ..).
// RT: 0 W DRAM address, 1 first W tile, 2 partner xy (MODE 1), 3 h DRAM address, 4 first h page, 5 identity address,
//     6 the partial ring's address (CB 2, the same on every core), 7 down chain heads (0: h only drained), 8 their
//     h_all address, 9 this core's h K tile (its pair), 10.. the heads' xy. With heads, h(v) ([MT] tiles) goes into
//     every head's h_all buffer v % hbuf at K tile `pair` once GO (sem 12, bumped by the down coordinator) allows,
//     then the heads' GATH (sem 8) is bumped.
// CT: TensorAccessorArgs(W), TensorAccessorArgs(h), TensorAccessorArgs(identity)
#include "api/dataflow/dataflow_api.h"
#include "kc_common.hpp"

using namespace kc;

void kernel_main() {
    constexpr auto wa = TensorAccessorArgs<0>();
    constexpr auto ha = TensorAccessorArgs<wa.next_compile_time_args_offset()>();
    constexpr auto ia = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    const auto ws = TensorAccessor(wa, get_arg_val<uint32_t>(0), get_tile_size(W_CB));
    const uint32_t w_first = get_arg_val<uint32_t>(1);
    const uint32_t partner = get_arg_val<uint32_t>(2);
    const auto hs = TensorAccessor(ha, get_arg_val<uint32_t>(3), get_tile_size(H_CB));
    const uint32_t h_first = get_arg_val<uint32_t>(4);
    if constexpr (!W_STREAM) {
        cb_reserve_back(W_CB, KT * WC);
        uint32_t l1 = get_write_ptr(W_CB);
        for (uint32_t t = 0; t < KT * WC; ++t) {
            noc_async_read_page(w_first + t, ws, l1);
            l1 += get_tile_size(W_CB);
        }
        noc_async_read_barrier();
        cb_push_back(W_CB, KT * WC);
    }
    if constexpr (MODE != 0) {  // the identity tiles [[I, 0], [0, I]] (the partials' add into DEST)
        const auto is = TensorAccessor(ia, get_arg_val<uint32_t>(5), get_tile_size(ID_CB));
        cb_reserve_back(ID_CB, 4);
        uint32_t l1 = get_write_ptr(ID_CB);
        for (uint32_t t = 0; t < 4; ++t) {
            noc_async_read_page(t, is, l1);
            l1 += get_tile_size(ID_CB);
        }
        noc_async_read_barrier();
        cb_push_back(ID_CB, 4);
    }
    volatile tt_l1_ptr uint32_t* xarr = kc_sem(XARR);
    volatile tt_l1_ptr uint32_t* xcons_w = kc_sem(XCONS);
    volatile tt_l1_ptr uint32_t* parr = kc_sem(PARR);
    volatile tt_l1_ptr uint32_t* pfree = kc_sem(PFREE);
    const uint64_t p_ring = MODE ? kc_xy(partner, get_arg_val<uint32_t>(6)) : 0;
    const uint64_t p_parr = MODE ? kc_xy(partner, get_semaphore(PARR)) : 0;
    const uint64_t p_pfree = MODE ? kc_xy(partner, get_semaphore(PFREE)) : 0;
    const uint32_t nheads = get_arg_val<uint32_t>(7), h_all = get_arg_val<uint32_t>(8), h_pair = get_arg_val<uint32_t>(9);
    constexpr uint32_t HBUF = get_named_compile_time_arg_val("hbuf");
    constexpr uint32_t H_TILE_B = get_named_compile_time_arg_val("x_tile_bytes");  // (h bfp8, as x)
    constexpr uint32_t NGU = get_named_compile_time_arg_val("ngu");
    const uint32_t h_buf_bytes = NGU * MT * H_TILE_B;
    volatile tt_l1_ptr uint32_t* go = kc_sem(12);
    uint32_t x_pub = 0, x_cons = 0;
    uint32_t p_sent = 0, p_pending = 0, p_pub = 0, p_cons = 0, h_done = 0;
    const uint32_t p_need = MODE ? NSB : 0;
    while (h_done < NSB || p_sent < p_need || p_cons < p_need || x_cons < TOTAL_BLOCKS) {
        invalidate_l1_cache();
        // x
        const uint32_t arrived = *xarr / XP;
        while (arrived > x_pub) {
            cb_push_back(X_CB, X_BLK);
            ++x_pub;
        }
        if (x_cons < x_pub && cb_pages_reservable_at_back(X_CB, (XS - (x_pub - x_cons) + 1) * X_BLK)) {
            ++x_cons;
            *xcons_w = x_cons;
        }
        if constexpr (MODE != 0) {
            // partials out
            if (!p_pending && p_sent < NSB && cb_pages_available_at_front(POUT_CB, P_TILES) &&
                (p_sent < PS || *pfree >= p_sent + 1 - PS)) {
                noc_async_write(get_read_ptr(POUT_CB), p_ring + (p_sent % PS) * P_BYTES, P_BYTES);
                p_pending = 1;
            }
            if (p_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
                noc_semaphore_inc(p_parr, 1);
                cb_pop_front(POUT_CB, P_TILES);
                ++p_sent;
                p_pending = 0;
            }
            // partials in
            while (*parr > p_pub) {
                cb_push_back(PIN_CB, P_TILES);
                ++p_pub;
            }
            if (p_cons < p_pub && cb_pages_reservable_at_back(PIN_CB, (PS - (p_pub - p_cons) + 1) * P_TILES)) {
                ++p_cons;
                noc_semaphore_inc(p_pfree, 1);
            }
        }
        // h
        if (h_done < NSB && cb_pages_available_at_front(H_CB, MT) && (nheads == 0 || *go + HBUF >= h_done + 1)) {
            const uint32_t src = get_read_ptr(H_CB);
            if (h_done + 1 == NSB) {
                uint32_t l1 = src;
                for (uint32_t t = 0; t < MT; ++t) {
                    noc_async_write_page(h_first + t, hs, l1);
                    l1 += get_tile_size(H_CB);
                }
            }
            for (uint32_t hd = 0; hd < nheads; ++hd) {
                noc_async_write(
                    src,
                    kc_xy(get_arg_val<uint32_t>(10 + hd), h_all + (h_done % HBUF) * h_buf_bytes + h_pair * MT * H_TILE_B),
                    MT * H_TILE_B);
            }
            noc_async_write_barrier();
            for (uint32_t hd = 0; hd < nheads; ++hd) {
                noc_semaphore_inc(kc_xy(get_arg_val<uint32_t>(10 + hd), get_semaphore(8)), 1);
            }
            cb_pop_front(H_CB, MT);
            ++h_done;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_full_barrier();
}
