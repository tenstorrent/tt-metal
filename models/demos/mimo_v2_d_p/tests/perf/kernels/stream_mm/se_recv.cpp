// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Streamed-expert receiver data movement (BRISC) on a compute core: a non-blocking event loop that
//   * publishes the resident x (pre-blocked in its CB) once per expert,
//   * runs the in1 landing ring (credits to the forwarder per free slot, publishes blocks as they land),
//   * exchanges h without ever blocking the ring, as gather + broadcast: this core's slice (MT tiles, k-tile MY_IDX)
//     is written into the coordinator's h_all (row-major within each K-block, half e % 2) and acknowledged with a
//     GATH increment there; once all NCC slices of an expert have arrived the coordinator multicasts that h_all half
//     to the whole worker grid (one sender, so concurrent overlapping multicasts can't deadlock; non-peers just absorb
//     the bytes at h_all's reserved address) and bumps every peer's arrival counter for that half,
//   * drains each expert's output and reports it to the coordinator (compute core 0), which counts the experts every
//     core has finished ("go"). h_all is double-buffered by expert parity: the slice of expert e goes out once
//     go >= e - 1 (every core has finished expert e - 2, the previous user of that half).
//
// CT: 0 X_CB, 1 X_TILES, 2 IN1_CB, 3 BLK_TILES, 4 BLOCKS_PER_EXPERT, 5 NUM_EXPERTS, 6 SLOTS, 7 OUT_CB, 8 OUT_TILES,
//     9 H_LOCAL_CB, 10 H_ALL_CB, 11 MT, 12 H_TILE_BYTES, 13 NCC, 14 DATA_SEM, 15 HARR_SEM, 16 GO_SEM, 17 DONE_SEM,
//     18 KBLK, 19 HARR1_SEM (arrival counter of the odd half; HARR_SEM counts the even half),
//     20 COMPUTE_ONLY (profiling mode: no weight stream / h exchange; blocks are republished from L1 as-is),
//     21 GATH_SEM (coordinator: h slices received), 22 HBUF (h_all buffers: 2 = by expert parity, 1 = single; the
//     slice of expert e then waits for go >= e, i.e. every core done with down(e - 1))
// RT: 0 forwarder xy, 1 credit sem id on the forwarder, 2 my index, 3 h_all base address, 4 coordinator xy,
//     5.. NCC compute-core xy (packed x << 16 | y), then NUM_RECTS and per rectangle: start xy, end xy (NOC0 order),
//     destinations excluding self, unused
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_ZONES
#include "tools/profiler/kernel_profiler.hpp"
#define SE_MARK(name)            \
    {                            \
        DeviceZoneScopedN(name); \
    }
#else
#define SE_MARK(name)
#endif

void kernel_main() {
    constexpr uint32_t x_cb = get_compile_time_arg_val(0);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t in1_cb = get_compile_time_arg_val(2);
    constexpr uint32_t blk_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t blocks_per_expert = get_compile_time_arg_val(4);
    constexpr uint32_t num_experts = get_compile_time_arg_val(5);
    constexpr uint32_t slots = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t out_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t h_local_cb = get_compile_time_arg_val(9);
    constexpr uint32_t h_all_cb = get_compile_time_arg_val(10);
    constexpr uint32_t mt = get_compile_time_arg_val(11);
    constexpr uint32_t h_tile_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t ncc = get_compile_time_arg_val(13);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(14);
    constexpr uint32_t harr_sem_id = get_compile_time_arg_val(15);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(16);
    constexpr uint32_t done_sem_id = get_compile_time_arg_val(17);
    constexpr uint32_t kblk = get_compile_time_arg_val(18);
    constexpr uint32_t harr1_sem_id = get_compile_time_arg_val(19);
    constexpr bool compute_only = get_compile_time_arg_val(20) != 0;
    constexpr uint32_t gath_sem_id = get_compile_time_arg_val(21);
    constexpr uint32_t hbuf = get_compile_time_arg_val(22);  // h_all halves: 2 (double-buffered) or 1
    static_assert(hbuf == 1 || hbuf == 2);
    constexpr uint32_t total_blocks = blocks_per_expert * num_experts;
    constexpr uint32_t h_all_tiles = ncc * mt;

    const uint32_t fxy = get_arg_val<uint32_t>(0);
    const uint32_t credit_sem_id = get_arg_val<uint32_t>(1);
    const uint32_t my_idx = get_arg_val<uint32_t>(2);
    const uint32_t h_all_addr = get_arg_val<uint32_t>(3);
    const uint32_t cxy = get_arg_val<uint32_t>(4);
    const bool is_coord = my_idx == 0;
    const uint64_t credit_noc = get_noc_addr(fxy >> 16, fxy & 0xFFFF, get_semaphore(credit_sem_id));
    const uint64_t done_noc = get_noc_addr(cxy >> 16, cxy & 0xFFFF, get_semaphore(done_sem_id));
    volatile tt_l1_ptr uint32_t* data_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_sem_id));
    volatile tt_l1_ptr uint32_t* done_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(done_sem_id));

    if constexpr (!compute_only) {
        noc_semaphore_inc(credit_noc, slots);  // every landing slot starts free
    }
    uint32_t granted = slots, pushed = 0;
    uint32_t x_pub = 0, h_pub = 0, out_done = 0, go_sent = 0;
    volatile tt_l1_ptr uint32_t* harr_sem[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(harr_sem_id)),
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(harr1_sem_id))};
    volatile tt_l1_ptr uint32_t* go_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id));
    // h send state: 0 idle, 2 waiting for the slice writes' acks. Broadcast state (coordinator): 0 idle, 1 issuing
    // multicast pieces, 2 waiting for their acks.
    const uint32_t num_rects = get_arg_val<uint32_t>(5 + ncc);
    constexpr uint32_t max_rects = 4;
    uint64_t rect_noc[max_rects];
    uint32_t rect_dests[max_rects];
    for (uint32_t r = 0; r < num_rects; ++r) {
        const uint32_t a0 = get_arg_val<uint32_t>(6 + ncc + r * 4), a1 = get_arg_val<uint32_t>(7 + ncc + r * 4);
        rect_noc[r] = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        rect_dests[r] = get_arg_val<uint32_t>(8 + ncc + r * 4);
    }
    constexpr uint32_t half_bytes = h_all_tiles * h_tile_bytes;
#ifndef SE_BC_PIECE
#define SE_BC_PIECE 16384  // one NoC burst; larger pieces block BRISC longer and starve the weight ring
#endif
#ifndef SE_BC_PER_PASS
#define SE_BC_PER_PASS 1
#endif
    constexpr uint32_t piece = SE_BC_PIECE;
    constexpr uint32_t pieces_per_pass = SE_BC_PER_PASS;
    const uint32_t slot_off = ((my_idx / kblk) * mt * kblk + my_idx % kblk) * h_tile_bytes;
    const uint64_t coord_h_all = get_noc_addr(cxy >> 16, cxy & 0xFFFF, h_all_addr);
    const uint64_t gath_noc = get_noc_addr(cxy >> 16, cxy & 0xFFFF, get_semaphore(gath_sem_id));
    volatile tt_l1_ptr uint32_t* gath_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gath_sem_id));
    uint32_t h_sent = 0, h_state = 0;
    uint32_t bc_n = 0, bc_state = 0, bc_off = 0, bc_r = 0;

    if constexpr (compute_only) {
        // Feed compute as fast as it can consume: whatever is in the landing ring / h_all is reused unchanged.
        while (out_done < num_experts) {
            if (x_pub < num_experts && cb_pages_reservable_at_back(x_cb, x_tiles)) {
                cb_push_back(x_cb, x_tiles);
                ++x_pub;
            }
            if (pushed < total_blocks && cb_pages_reservable_at_back(in1_cb, blk_tiles)) {
                cb_push_back(in1_cb, blk_tiles);
                ++pushed;
            }
            if (cb_pages_available_at_front(h_local_cb, mt)) {
                cb_pop_front(h_local_cb, mt);
            }
            if (h_pub < num_experts && cb_pages_reservable_at_back(h_all_cb, h_all_tiles)) {
                cb_push_back(h_all_cb, h_all_tiles);
                ++h_pub;
            }
            if (cb_pages_available_at_front(out_cb, out_tiles)) {
                if (out_done + 1 < num_experts) {
                    cb_pop_front(out_cb, out_tiles);
                }
                ++out_done;
            }
        }
        return;
    }
    while (out_done < num_experts) {
        invalidate_l1_cache();
        if (x_pub < num_experts && cb_pages_reservable_at_back(x_cb, x_tiles)) {
            cb_push_back(x_cb, x_tiles);  // same data, same memory: compute is done with the previous expert's x
            ++x_pub;
        }
        if (granted < total_blocks && cb_pages_reservable_at_back(in1_cb, (granted - pushed + 1) * blk_tiles)) {
            noc_semaphore_inc(credit_noc, 1);
            ++granted;
        }
        if (*data_sem > pushed) {
            cb_push_back(in1_cb, blk_tiles);
            ++pushed;
        }
        if (h_state == 0 && h_sent < num_experts && *go_sem + hbuf >= h_sent + 1 &&
            cb_pages_available_at_front(h_local_cb, mt)) {
            const uint32_t src = get_read_ptr(h_local_cb);
            const uint32_t dst = (h_sent % hbuf) * half_bytes + slot_off;
            for (uint32_t m = 0; m < mt; ++m) {
                noc_async_write(src + m * h_tile_bytes, coord_h_all + dst + m * kblk * h_tile_bytes, h_tile_bytes);
            }
            h_state = 2;
            SE_MARK("SE_HSEND");
        }
        if (h_state == 2 && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            noc_semaphore_inc(gath_noc, 1);
            SE_MARK("SE_HFLUSH");
            cb_pop_front(h_local_cb, mt);
            ++h_sent;
            h_state = 0;
        }
        if (is_coord) {
            if (bc_state == 0 && bc_n < num_experts && *gath_sem >= ncc * (bc_n + 1)) {
                bc_off = 0;
                bc_r = 0;
                bc_state = 1;
                SE_MARK("SE_BC_START");
            }
            for (uint32_t n = 0; n < pieces_per_pass && bc_state == 1; ++n) {
                const uint32_t src = h_all_addr + (bc_n % hbuf) * half_bytes + bc_off;
                const uint32_t len = half_bytes - bc_off < piece ? half_bytes - bc_off : piece;
                if (rect_dests[bc_r] != 0) {
                    noc_async_write_multicast(src, rect_noc[bc_r] | src, len, rect_dests[bc_r]);
                }
                if (++bc_r == num_rects) {
                    bc_r = 0;
                    bc_off += len;
                    if (bc_off == half_bytes) {
                        bc_state = 2;
                    }
                }
            }
            if (bc_state == 2 && ncrisc_noc_nonposted_writes_flushed(noc_index)) {
                const uint32_t arr_sem = get_semaphore((bc_n % hbuf) ? harr1_sem_id : harr_sem_id);
                for (uint32_t p = 0; p < ncc; ++p) {
                    const uint32_t xy = get_arg_val<uint32_t>(5 + p);
                    noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, arr_sem), 1);
                }
                ++bc_n;
                bc_state = 0;
                SE_MARK("SE_BC_DONE");
            }
        }
        if (h_pub < num_experts && *harr_sem[h_pub % hbuf] >= h_pub / hbuf + 1 &&
            cb_pages_reservable_at_back(h_all_cb, h_all_tiles)) {
            cb_push_back(h_all_cb, h_all_tiles);
            ++h_pub;
            SE_MARK("SE_HPUB");
        }
        if (cb_pages_available_at_front(out_cb, out_tiles)) {
            if (out_done + 1 < num_experts) {
                cb_pop_front(out_cb, out_tiles);  // the output shard keeps the last expert's result
            }
            noc_semaphore_inc(done_noc, 1);
            ++out_done;
        }
        if (is_coord && go_sent + 1 < num_experts && *done_sem >= ncc * (go_sent + 1)) {
            for (uint32_t p = 0; p < ncc; ++p) {
                const uint32_t xy = get_arg_val<uint32_t>(5 + p);
                noc_semaphore_inc(get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(go_sem_id)), 1);
            }
            ++go_sent;
        }
    }
    noc_async_atomic_barrier();
}
