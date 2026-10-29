// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident MLP streamer writer (BRISC): the activation plumbing of one streamer core.
//   - releases the hub-multicast partial slots of the layer to compute,
//   - sends this core's slice of silu(g) * u to every streamer (itself included) and releases the full
//     intermediate activation to compute once all slices have landed,
//   - sends this core's down-projection columns to the hub's slot of this chip.
// No buffer needs double-buffering: each one is only rewritten after a dependency chain that goes
// through this core's consumption of it (e.g. a peer's next slice needs the hub's next slots, which
// need this core's down output, which needs this core to be done with the current activation).
//
// Runtime args: 0 ng (gate tiles of this core), 1 act_off (tile offset of its slice), 2 nd (down
// tiles), 3 pout_off (tile offset of its down columns), 4 hub_noc_x, 5 hub_noc_y, 6 hub_slots_addr,
// 7 act_addr (cb_act tensor base, same on every streamer), then num_streamers x (noc_x, noc_y).

#include "api/dataflow/dataflow_api.h"
#include "mlp_common.hpp"

using namespace resident_mlp;

void kernel_main() {
    const uint32_t ng = get_arg_val<uint32_t>(0);
    const uint32_t act_off = get_arg_val<uint32_t>(1);
    const uint32_t nd = get_arg_val<uint32_t>(2);
    const uint32_t pout_off = get_arg_val<uint32_t>(3);
    const uint32_t hub_x = get_arg_val<uint32_t>(4);
    const uint32_t hub_y = get_arg_val<uint32_t>(5);
    const uint32_t hub_slots = get_arg_val<uint32_t>(6);
    const uint32_t act_addr = get_arg_val<uint32_t>(7);
    constexpr uint32_t peers_base = 8;

    const uint32_t slots_sem_addr = get_semaphore(sem_slots);
    const uint32_t act_sem_addr = get_semaphore(sem_act);
    const uint32_t gather_sem_addr = get_semaphore(sem_gather);
    volatile tt_l1_ptr uint32_t* slots_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slots_sem_addr);
    volatile tt_l1_ptr uint32_t* act_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(act_sem_addr);
    const uint64_t hub_gather_sem = get_noc_addr(hub_x, hub_y, gather_sem_addr);

    for (uint32_t l = 0; l < layers; l++) {
        // partial slots of this layer (layer 0: preloaded by the host)
        if (l > 0) {
            noc_semaphore_wait_min(slots_sem, l);
        }
        cb_reserve_back(cb_slots, num_chips * Ht);
        cb_push_back(cb_slots, num_chips * Ht);

        // intermediate activation exchange
        cb_wait_front(cb_aslice, ng_max);
        const uint32_t slice = get_read_ptr(cb_aslice);
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_async_write(slice, get_noc_addr(px, py, act_addr + act_off * kTileBytes), ng * kTileBytes);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < num_streamers; s++) {
            const uint32_t px = get_arg_val<uint32_t>(peers_base + 2 * s);
            const uint32_t py = get_arg_val<uint32_t>(peers_base + 2 * s + 1);
            noc_semaphore_inc(get_noc_addr(px, py, act_sem_addr), 1);
        }
        cb_pop_front(cb_aslice, ng_max);
        noc_semaphore_wait_min(act_sem, num_streamers * (l + 1));
        cb_reserve_back(cb_act, It);
        cb_push_back(cb_act, It);

        // down-projection columns -> this chip's slot on the hub (parity ping-pong for the fabric)
        cb_wait_front(cb_pout, nd_max);
        const uint32_t dst = hub_slots + (((l & 1) * num_chips + chip) * Ht + pout_off) * kTileBytes;
        noc_async_write(get_read_ptr(cb_pout), get_noc_addr(hub_x, hub_y, dst), nd * kTileBytes);
        noc_async_write_barrier();
        noc_semaphore_inc(hub_gather_sem, 1);
        cb_pop_front(cb_pout, nd_max);
    }
    noc_async_atomic_barrier();
}
