// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert: a bank reader that also computes some down columns (BRISC, NOC0). One non-blocking loop does
//   * the reader's job (se_reader.cpp): its gate/up weight region, chunk by chunk, into the forwarder's CB,
//   * the down weight reads for its own down columns (se6_dw.cpp) into the down compute's in1 ring,
//   * a down-chain tail's h handling (se6_drecv.cpp with no successor): h(v) arrives from the chain predecessor in
//     H_PIECES pieces (HARR), goes to compute once complete, and consumed h is reported back (HSFREE, atomic delta),
//   * the output: [MT x PCD] bf16 tiles of each virtual expert to y in DRAM, then a done report to the coordinator.
// Reads of both weight streams share one read barrier per loop pass (a pass issues at most one chunk of each).
//
// CT: 0 GU_CB, 1 W_TILE_BYTES, 2 GU_SLOT_TILES, 3 GU_CHUNKS (all experts), 4 D_CB (in1), 5 D_SLOT_TILES, 6 D_BLOCKS
//     (all experts), 7 H_ALL_CB, 8 H_ALL_TILES, 9 H_TILE_BYTES, 10 H_PIECES, 11 OUT_CB, 12 OUT_TILES, 13 NUM_V,
//     14 HARR_SEM, 15 HSFREE_SEM, 16 HBUF, 17 MT, 18 PCD, 19 HT, 20 S (sub-blocks per expert)
// RT: 0 gu weight bank base, 1 gu bank id, 2 gu region offset, 3 down weight bank base, 4 down bank id, 5 down region
//     offset, 6 predecessor xy, 7 coordinator xy, 8 done words address (coordinator), 9 this core's done index,
//     10 y DRAM address, 11 first y tile column
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif

void kernel_main() {
    constexpr uint32_t gu_cb = get_compile_time_arg_val(0);
    constexpr uint32_t w_tile = get_compile_time_arg_val(1);
    constexpr uint32_t gu_slot = get_compile_time_arg_val(2);
    constexpr uint32_t gu_chunks_ct = get_compile_time_arg_val(3);
    constexpr uint32_t d_cb = get_compile_time_arg_val(4);
    constexpr uint32_t d_slot = get_compile_time_arg_val(5);
    constexpr uint32_t d_blocks_ct = get_compile_time_arg_val(6);
    constexpr uint32_t h_all_cb = get_compile_time_arg_val(7);
    constexpr uint32_t h_all_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t h_pieces = get_compile_time_arg_val(10);
    constexpr uint32_t out_cb = get_compile_time_arg_val(11);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(12);
    constexpr uint32_t num_v_ct = get_compile_time_arg_val(13);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t hsfree_sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t hbuf = get_compile_time_arg_val(16);
    constexpr uint32_t mt = get_compile_time_arg_val(17);
    constexpr uint32_t pcd = get_compile_time_arg_val(18);
    constexpr uint32_t ht = get_compile_time_arg_val(19);
    constexpr uint32_t sub = get_compile_time_arg_val(20);
    constexpr uint32_t gu_bytes = gu_slot * w_tile;
    constexpr uint32_t d_bytes = d_slot * w_tile;

    uint64_t gu_src =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(0) + get_arg_val<uint32_t>(2));
    uint64_t d_src =
        get_noc_addr_from_bank_id<true>(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(3) + get_arg_val<uint32_t>(5));
    const uint32_t pxy = get_arg_val<uint32_t>(6), cxy = get_arg_val<uint32_t>(7);
    const uint64_t pred_free = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(hsfree_sem_id));
    const uint64_t done_noc =
        get_noc_addr(cxy >> 16, cxy & 0xFFFF, get_arg_val<uint32_t>(8) + 4 * get_arg_val<uint32_t>(9));
#ifdef SE_E2E
    // y is the bfp8 tile buffer shaped like the dispatch buffer: virtual expert v = (e, s) writes row tiles
    // region_tile(e) + s * MT + r, only those below the expert's token count (RT: per expert region tile, count tiles)
    constexpr uint32_t out_page = 1088;
    const InterleavedAddrGenFast<true> y = {
        .bank_base_address = get_arg_val<uint32_t>(10), .page_size = out_page, .data_format = DataFormat::Bfp8_b};
    const uint32_t e2e_args = 12;
#else
    constexpr uint32_t out_page = 2048;
    const InterleavedAddrGenFast<true> y = {
        .bank_base_address = get_arg_val<uint32_t>(10), .page_size = 2048, .data_format = DataFormat::Float16_b};
#endif
    const uint32_t col0 = get_arg_val<uint32_t>(11);
    volatile tt_l1_ptr uint32_t* harr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(harr_sem_id));

#ifdef SE_DYN
    // Dynamic counts (with SE_E2E): only the active experts' weights and sub-blocks; RT 12.. are the se_dyn.hpp args
    // (CT 21 NUM_E), CB 7 this RISC's scratch; this core's down compute gets them in CB 6.
    constexpr uint32_t num_e = get_compile_time_arg_val(21);
    constexpr uint32_t gu_per_e = gu_chunks_ct / num_e, d_per_e = d_blocks_ct / num_e;
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 12, get_write_ptr(tt::CBIndex::c_7), mt * 32);
    se_dyn_publish(dyn, tt::CBIndex::c_6);
    const uint32_t gu_chunks = dyn.n_act * gu_per_e, d_blocks = dyn.n_act * d_per_e, num_v = dyn.num_v;
    const uint64_t gu_base = gu_src, d_base = d_src;
    uint32_t y_a = 0, y_s = 0;
#else
    constexpr uint32_t gu_chunks = gu_chunks_ct, d_blocks = d_blocks_ct, num_v = num_v_ct;
#endif
    uint32_t gu_read = 0, d_read = 0;
    uint32_t h_pub = 0, h_cons = 0, h_rep = 0, out_done = 0;
    bool y_pending = false;
    while (gu_read < gu_chunks || d_read < d_blocks || out_done < num_v) {
        invalidate_l1_cache();
        // weights: at most one chunk of each stream per pass, one barrier for both
        const bool rd_gu = gu_read < gu_chunks && cb_pages_reservable_at_back(gu_cb, gu_slot);
#ifdef SE_GU_FIRST
        // gate/up weights first: a down block only when the forwarder's CB is full (they are needed much later)
        const bool rd_d = !rd_gu && d_read < d_blocks && cb_pages_reservable_at_back(d_cb, d_slot);
#else
        const bool rd_d = d_read < d_blocks && cb_pages_reservable_at_back(d_cb, d_slot);
#endif
#ifdef SE_DYN
        if (rd_gu) {
            gu_src = gu_base + (dyn.eid[gu_read / gu_per_e] * gu_per_e + gu_read % gu_per_e) * gu_bytes;
        }
        if (rd_d) {
            d_src = d_base + (dyn.eid[d_read / d_per_e] * d_per_e + d_read % d_per_e) * d_bytes;
        }
#endif
        if (rd_gu) {
            noc_async_read(gu_src, get_write_ptr(gu_cb), gu_bytes);
            gu_src += gu_bytes;
        }
        if (rd_d) {
            noc_async_read(d_src, get_write_ptr(d_cb), d_bytes);
            d_src += d_bytes;
        }
        if (rd_gu || rd_d) {
            noc_async_read_barrier();
            if (rd_gu) {
                cb_push_back(gu_cb, gu_slot);
                ++gu_read;
            }
            if (rd_d) {
                cb_push_back(d_cb, d_slot);
                ++d_read;
            }
        }
        // h from the chain predecessor
        if (h_pub < num_v && *harr >= (h_pub + 1) * h_pieces && h_pub - h_cons < hbuf) {
            cb_push_back(h_all_cb, h_all_tiles);
            ++h_pub;
        }
        if (h_cons < h_pub && cb_pages_reservable_at_back(h_all_cb, (hbuf - (h_pub - h_cons) + 1) * h_all_tiles)) {
            ++h_cons;
        }
        if (h_cons > h_rep) {
            noc_semaphore_inc(pred_free, h_cons - h_rep);
            h_rep = h_cons;
        }
        // output
        if (!y_pending && out_done < num_v && cb_pages_available_at_front(out_cb, out_tiles)) {
            const uint32_t src = get_read_ptr(out_cb);
            for (uint32_t r = 0; r < mt; ++r) {
#if defined(SE_DYN)
                if (y_s * mt * 32 + r * 32 >= dyn.cnt[y_a]) {
                    continue;
                }
                const uint32_t trow = dyn.off[y_a] / 32 + y_s * mt + r;
#elif defined(SE_E2E)
                const uint32_t e = out_done / sub, s_ = out_done % sub;
                if (s_ * mt + r >= get_arg_val<uint32_t>(e2e_args + 2 * e + 1)) {
                    continue;
                }
                const uint32_t trow = get_arg_val<uint32_t>(e2e_args + 2 * e) + s_ * mt + r;
#else
                const uint32_t trow = out_done * mt + r;
#endif
                for (uint32_t c = 0; c < pcd; ++c) {
                    noc_async_write_tile(trow * ht + col0 + c, y, src + (r * pcd + c) * out_page);
                }
            }
            y_pending = true;
        }
        if (y_pending && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            cb_pop_front(out_cb, out_tiles);
            noc_semaphore_inc(done_noc, 1);
            ++out_done;
            y_pending = false;
#ifdef SE_DYN
            if (++y_s == dyn.subs[y_a]) {
                y_s = 0;
                ++y_a;
            }
#endif
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
