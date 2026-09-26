// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Spatially pipelined expert: a down core's data movement (BRISC, NOC0), a non-blocking event loop that
//   * receives h of every virtual expert: a chain head gathers it directly (every gate/up core writes its slice here,
//     GATH counts the slices), other cores get it from their chain predecessor in H_PIECES pieces (HARR counts
//     pieces); each piece is forwarded to the successor as soon as it is here and the successor has freed its h_all
//     (HSFREE, in whole h), up to LINK_DEPTH pieces in flight (NoC transaction ids); h_all(v) goes to compute once
//     complete, and min(consumed, forwarded) h is reported to the predecessor (atomic delta),
//   * writes each output block ([MT x PCD] bf16 tiles of virtual expert v) to y in DRAM (interleaved, tile (row, col) =
//     page row * HT + col), frees its slot once the writes are acknowledged, and reports it to the coordinator,
//   * on the coordinator: counts the virtual experts every down core has finished and releases the gate/up cores'
//     next h ("go": a gate/up core writes its slice of h(v) once every down core is done with down(v - 1), which
//     implies every chain head has consumed and forwarded h(v - 1)).
//
// CT: 0 H_ALL_CB, 1 H_ALL_TILES, 2 H_TILE_BYTES, 3 H_PIECES, 4 OUT_CB, 5 OUT_TILES, 6 NUM_V, 7 S, 8 N_SLICES (gate/up
//     cores), 9 N_DOWN, 10 HARR_SEM, 11 HSFREE_SEM, 12 GATH_SEM, 13 DONE_SEM, 14 GO_SEM, 15 HBUF (h_all buffers: h(v)
//     lives in buffer v % HBUF; a gate/up core writes its slice of h(v) once every down core is done with down(v -
//     HBUF)) 16 MT (row tiles per virtual expert), 17 PCD, 18 HT, 19-21 GATH sem ids of h buffers 0..2 (slices of h(v)
//     are counted per buffer v % HBUF: one shared count could be completed by a fast core's next slice)
// RT: 0 h_all address, 1 predecessor xy (0: head), 2 successor xy (0: tail), 3 coordinator xy, 4 is coordinator,
//     5 N_GU, 6 y DRAM address, 7 first y tile column, 8 address of the coordinator's per-down-core done words, 9 this
//     core's index, 10.. N_GU gate/up core xy. "go" is released on the minimum of the done words (a summed count would
//     let fast down cores stand in for a slow one); the coordinator zeroes them again at the end.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif

void kernel_main() {
    constexpr uint32_t h_all_cb = get_compile_time_arg_val(0);
    constexpr uint32_t h_all_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t h_pieces = get_compile_time_arg_val(3);
    constexpr uint32_t out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t num_v_ct = get_compile_time_arg_val(6);
    constexpr uint32_t sub = get_compile_time_arg_val(7);
    constexpr uint32_t n_slices = get_compile_time_arg_val(8);
    constexpr uint32_t n_down = get_compile_time_arg_val(9);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t hsfree_sem_id = get_compile_time_arg_val(11);
    constexpr uint32_t gath_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(13);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t hbuf = get_compile_time_arg_val(15);
    constexpr uint32_t mt = get_compile_time_arg_val(16);
    constexpr uint32_t pcd = get_compile_time_arg_val(17);
    constexpr uint32_t ht = get_compile_time_arg_val(18);
    static_assert(hbuf <= 3);
    const uint32_t gath_ids[3] = {
        get_compile_time_arg_val(19), get_compile_time_arg_val(20), get_compile_time_arg_val(21)};
    constexpr uint32_t h_bytes = h_all_tiles * h_tile_bytes;
    constexpr uint32_t piece_bytes = h_bytes / h_pieces;
    constexpr uint32_t link_depth = 3;

    const uint32_t h_all = get_arg_val<uint32_t>(0);
    const uint32_t pxy = get_arg_val<uint32_t>(1), sxy = get_arg_val<uint32_t>(2), cxy = get_arg_val<uint32_t>(3);
    const bool is_coord = get_arg_val<uint32_t>(4) != 0;
    const uint32_t n_gu = get_arg_val<uint32_t>(5);
#ifdef SE_E2E
    // y is the bfp8 tile buffer shaped like the dispatch buffer: virtual expert v = (e, s) writes row tiles
    // region_tile(e) + s * MT + r, only those below the expert's token count (RT: per expert region tile, count tiles)
    constexpr uint32_t out_page = 1088;
    const InterleavedAddrGenFast<true> y = {
        .bank_base_address = get_arg_val<uint32_t>(6), .page_size = out_page, .data_format = DataFormat::Bfp8_b};
    const uint32_t e2e_args = 10 + n_gu;
#endif
#ifdef SE_DYN
    // Dynamic counts (with SE_E2E): the active experts' sub-blocks; RT 10 + N_GU.. are the se_dyn.hpp args (CT 22
    // NUM_E), CB 7 this RISC's scratch; the compute gets them in CB 6. Output rows come from the experts' regions.
    constexpr uint32_t num_e_dyn = get_compile_time_arg_val(22);
    SeDyn dyn;
    se_dyn_load<num_e_dyn>(dyn, 10 + n_gu, get_write_ptr(tt::CBIndex::c_7), mt * 32);
    se_dyn_publish(dyn, tt::CBIndex::c_6);
    const uint32_t num_v = dyn.num_v;
    uint32_t y_a = 0, y_s = 0;  // (active expert, sub-block) of output out_done
#else
    constexpr uint32_t num_v = num_v_ct;
#endif
#ifdef SE_E2E
#else
    constexpr uint32_t out_page = 2048;
    const InterleavedAddrGenFast<true> y = {
        .bank_base_address = get_arg_val<uint32_t>(6), .page_size = 2048, .data_format = DataFormat::Float16_b};
#endif
    const uint32_t col0 = get_arg_val<uint32_t>(7);
    const uint32_t done_words = get_arg_val<uint32_t>(8);
    const uint32_t my_index = get_arg_val<uint32_t>(9);
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    volatile tt_l1_ptr uint32_t* harr = sem(harr_sem_id);
    volatile tt_l1_ptr uint32_t* hsfree = sem(hsfree_sem_id);
    volatile tt_l1_ptr uint32_t* gath[3] = {sem(gath_ids[0]), sem(gath_ids[1]), sem(gath_ids[2])};
    volatile tt_l1_ptr uint32_t* done_sem = sem(done_sem_id);
    const uint64_t pred_free = get_noc_addr(pxy >> 16, pxy & 0xFFFF, get_semaphore(hsfree_sem_id));
    const uint64_t succ_harr = get_noc_addr(sxy >> 16, sxy & 0xFFFF, get_semaphore(harr_sem_id));
    const uint64_t succ_hall = get_noc_addr(sxy >> 16, sxy & 0xFFFF, h_all);
    const uint64_t done_noc = get_noc_addr(cxy >> 16, cxy & 0xFFFF, done_words + 4 * my_index);
    volatile tt_l1_ptr uint32_t* dw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_words);
    const bool head = pxy == 0;
    auto write_trid = [](uint32_t src, uint64_t dst, uint32_t bytes, uint32_t trid) {
        for (uint32_t o = 0; o < bytes; o += NOC_MAX_BURST_SIZE) {
            const uint32_t n = bytes - o < NOC_MAX_BURST_SIZE ? bytes - o : NOC_MAX_BURST_SIZE;
            noc_async_write_one_packet_with_trid(src + o, dst + o, n, trid);
        }
    };

    uint32_t h_pub = 0, h_cons = 0, h_iss = 0, h_fwd = 0, h_rep = 0, h_done = 0;
    uint32_t out_done = 0, go_sent = 0;
    bool y_pending = false;
    while (out_done < num_v || (sxy && h_fwd < num_v * h_pieces) || (is_coord && go_sent + 1 < num_v)) {
        invalidate_l1_cache();
        if (head) {  // h(v) is complete once all slices of it are in its buffer
            while (h_done < num_v && *gath[h_done % hbuf] >= n_slices * (h_done / hbuf + 1)) {
                ++h_done;
            }
        }
        const uint32_t arrived = head ? h_done * h_pieces : *harr;
        while (sxy && h_iss < arrived && h_iss - h_fwd < link_depth && *hsfree + hbuf >= h_iss / h_pieces + 1) {
            const uint32_t off = ((h_iss / h_pieces) % hbuf) * h_bytes + (h_iss % h_pieces) * piece_bytes;
            write_trid(h_all + off, succ_hall + off, piece_bytes, 1 + h_iss % link_depth);
            ++h_iss;
        }
        while (h_fwd < h_iss &&
               ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + h_fwd % link_depth)) {
            noc_semaphore_inc(succ_harr, 1);
            ++h_fwd;
        }
        if (h_pub < num_v && arrived >= (h_pub + 1) * h_pieces && h_pub - h_cons < hbuf) {
            cb_push_back(h_all_cb, h_all_tiles);
            ++h_pub;
        }
        if (h_cons < h_pub && cb_pages_reservable_at_back(h_all_cb, (hbuf - (h_pub - h_cons) + 1) * h_all_tiles)) {
            ++h_cons;
        }
        const uint32_t freed = sxy && h_fwd / h_pieces < h_cons ? h_fwd / h_pieces : h_cons;
        if (freed > h_rep) {
            if (!head) {
                noc_semaphore_inc(pred_free, freed - h_rep);
            }
            h_rep = freed;
        }
        if (!y_pending && out_done < num_v && cb_pages_available_at_front(out_cb, out_tiles)) {
            const uint32_t src = get_read_ptr(out_cb);
            for (uint32_t r = 0; r < mt; ++r) {
#ifdef SE_E2E
#ifdef SE_DYN
                if (y_s * mt * 32 + r * 32 >= dyn.cnt[y_a]) {
                    continue;
                }
                const uint32_t trow = dyn.off[y_a] / 32 + y_s * mt + r;
#else
                const uint32_t e = out_done / sub, s_ = out_done % sub;
                if (s_ * mt + r >= get_arg_val<uint32_t>(e2e_args + 2 * e + 1)) {
                    continue;
                }
                const uint32_t trow = get_arg_val<uint32_t>(e2e_args + 2 * e) + s_ * mt + r;
#endif
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
#ifdef SE_DYN
            if (++y_s == dyn.subs[y_a]) {
                y_s = 0;
                ++y_a;
            }
#endif
            y_pending = false;
        }
        bool all_done = is_coord && go_sent + 1 < num_v;
        for (uint32_t d = 0; all_done && d < n_down; ++d) {
            all_done = dw[d] >= go_sent + 1;
        }
        if (all_done) {
            for (uint32_t p = 0; p < n_gu; ++p) {
                const uint32_t xy = get_arg_val<uint32_t>(10 + p);
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(go_sem_id)), 1);
            }
            ++go_sent;
        }
    }
    if (is_coord) {  // every core's last report is in once all words reach NUM_V; leave them zeroed for the next run
        for (uint32_t d = 0; d < n_down; ++d) {
            while (true) {
                invalidate_l1_cache();
                if (dw[d] >= num_v) {
                    break;
                }
            }
        }
        for (uint32_t d = 0; d < n_down; ++d) {
            dw[d] = 0;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
