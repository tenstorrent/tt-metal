// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter ingest, streamer writer v2 (BRISC; M6b GI1 fix): d_ingest_writer.cpp (C1, frozen) with one
// k | v semaphore per layer on the slice (sem_kv + l). C1 bumped one cumulative sem_kv; a streamer's layer l + 1 k | v
// does not wait for the other streamers, so a fast streamer's layer l + 1 bump could stand in for a slow streamer's
// layer l bump, and the slice packed layer l before that streamer's tiles landed (GI1 diagnostics gi1_ringrep2.json).
//
// C1: Resident DFlash2 drafter ingest, streamer writer (BRISC; M6b C1), see d_ingest_common.hpp:
//   take the taps' tiles (every slice has multicast its K-tile range: sem_in >= K); the fc partial into its slice's
//   slot of this chip and the hub's staging (round 0), as d_streamer_writer.cpp's send_partial
//   take ctx (sem_in >= 2 K); per drafter layer this core's k | v tiles into its head block's slice (slice = bank):
//   the slice's k | v buffer at layer l's block, column offset 2 j; bump the slice's sem_kv
//   then one bump of the hub's record semaphore (see d_ingest_reader.cpp)
// Mailboxes as d_streamer_writer.cpp: streamer 0 publishes cb_in's address to the slices (sem_haddr), the hub its
// staging address (sem_stage), slice 0 the slots address (sem_slotaddr) and the k | v buffer address (sem_kvaddr).
//
// Runtime args: 0 nd, 1 pout_off (first hidden column tile), 2 hub_x, 3 hub_y, 4 hub sem_gather id, 5 hub sem_rec id,
// 6 core index, 7 slice_x, 8 slice_y (the slice owning this core's hidden columns, = its head block's slice),
// 9 sem_stage id, 10 / 11 the slice's slot semaphore (parity 0 / 1), 12 first column within the slice, 13 slot bytes,
// 14 sem_slotaddr id, 15 sem_haddr id (the slices'), 16 sem_kvaddr id, 17 j (k | v column offset 2 j in the block),
// then K x (x, y) of the slices

#include "api/dataflow/dataflow_api.h"
#include "d_ingest_common.hpp"

using namespace ding;

void kernel_main() {
    const uint32_t nd = get_arg_val<uint32_t>(0);
    const uint32_t pout_off = get_arg_val<uint32_t>(1);
    const uint32_t hub_x = get_arg_val<uint32_t>(2);
    const uint32_t hub_y = get_arg_val<uint32_t>(3);
    const uint32_t hub_gather = get_semaphore(get_arg_val<uint32_t>(4));
    const uint32_t hub_rec_sem = get_semaphore(get_arg_val<uint32_t>(5));
    const uint32_t core = get_arg_val<uint32_t>(6);
    const uint32_t slice_x = get_arg_val<uint32_t>(7);
    const uint32_t slice_y = get_arg_val<uint32_t>(8);
    volatile tt_l1_ptr uint32_t* stage_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(9)));
    const uint32_t slice_sem0 = get_arg_val<uint32_t>(10);
    const uint32_t slice_off = get_arg_val<uint32_t>(12);
    const uint32_t slot_bytes = get_arg_val<uint32_t>(13);
    volatile tt_l1_ptr uint32_t* slots_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(14)));
    const uint32_t haddr_box = get_semaphore(get_arg_val<uint32_t>(15));
    volatile tt_l1_ptr uint32_t* kvaddr_box =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(16)));
    const uint32_t j = get_arg_val<uint32_t>(17);
    constexpr uint32_t slices_base = 18;
    constexpr uint32_t num_chips = 4;

    auto cb_base = [](uint32_t cb) {
        auto& iface = get_local_cb_interface(cb);
        return iface.fifo_limit - iface.fifo_size;
    };
    if (core == 0) {
        for (uint32_t k = 0; k < K; k++) {
            const uint32_t sx = get_arg_val<uint32_t>(slices_base + 2 * k);
            const uint32_t sy = get_arg_val<uint32_t>(slices_base + 2 * k + 1);
            noc_inline_dw_write(get_noc_addr(sx, sy, haddr_box), cb_base(cb_in));
        }
    }
    volatile tt_l1_ptr uint32_t* in_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_in));

    // the taps' tiles
    noc_semaphore_wait_min(in_sem, K);
    cb_reserve_back(cb_in, Kin);
    cb_push_back(cb_in, Kin);
    // the fc partial (round 0: parity 0)
    cb_wait_front(cb_pout, nd_max);
    uint32_t hub_staging = 0, slice_slots = 0;
    while (hub_staging == 0 || slice_slots == 0) {
        invalidate_l1_cache();
        hub_staging = *stage_box;
        slice_slots = *slots_box;
    }
    {
        const uint32_t src = get_read_ptr(cb_pout);
        const uint32_t to_slice = slice_slots + chip * slot_bytes + slice_off * kTileBytes;
        noc_async_write(src, get_noc_addr(slice_x, slice_y, to_slice), nd * kTileBytes);
        noc_async_write(src, get_noc_addr(hub_x, hub_y, hub_staging + pout_off * kTileBytes), nd * kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(slice_x, slice_y, slice_sem0), 1);
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, hub_gather), 1);
        cb_pop_front(cb_pout, nd_max);
    }
    // ctx
    noc_semaphore_wait_min(in_sem, 2 * K);
    cb_reserve_back(cb_h, Ht);
    cb_push_back(cb_h, Ht);
    // per layer: k | v tiles into the head block's slice
    uint32_t slice_kv = 0;
    for (uint32_t l = 0; l < layers; l++) {
        cb_wait_front(cb_kv, kKvTiles);
        while (slice_kv == 0) {
            invalidate_l1_cache();
            slice_kv = *kvaddr_box;
        }
        const uint32_t dst = slice_kv + (l * kHeadTiles + j * kKvTiles) * kTileBytes;
        noc_async_write(get_read_ptr(cb_kv), get_noc_addr(slice_x, slice_y, dst), kKvTiles * kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(slice_x, slice_y, get_semaphore(sem_kv + l)), 1);
        cb_pop_front(cb_kv, kKvTiles);
    }
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, hub_rec_sem), 1);
    noc_async_atomic_barrier();
}
