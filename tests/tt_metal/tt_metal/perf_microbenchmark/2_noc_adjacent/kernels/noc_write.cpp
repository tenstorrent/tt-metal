// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

// STATE-BASED WRITES, one NOC packet each.
//
// Where the last 14% goes. NOC_MAX_BURST_SIZE is 8 KB (NOC_MAX_BURST_WORDS 256 x
// NOC_WORD_BYTES 32), so the shipped kernel's 64 KB noc_async_write is EIGHT
// bursts, and each one reprograms the NIU command registers from scratch. That
// per-burst issue cost -- not latency and not the barrier -- is what holds the
// measurement at 27.4 B/cc against a 32 B/cc clamp.
//
// Ruled out first, both measured, so this is not a guess:
//   * the barrier: batching 1/2/4/8 outstanding writes moved nothing (27.398,
//     27.399, 27.402, 27.391 B/cc).
//   * packet headers: one 32 B word per 8 KB burst is 0.39%, not 14%.
//
// set_state programs destination, size and VC once; with_state then issues each
// packet writing only the two addresses. It is what cq_dispatch uses for the
// same reason.
void kernel_main() {
    uint32_t adjacent_noc_x = get_arg_val<uint32_t>(0);
    uint32_t adjacent_noc_y = get_arg_val<uint32_t>(1);
    uint32_t adjacent_noc_cb1 = get_arg_val<uint32_t>(2);
    uint32_t iter_count = get_arg_val<uint32_t>(3);
    uint32_t tiles_per_transfer = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb0_id = 0;
    uint32_t transfer_size = get_tile_size(cb0_id) * tiles_per_transfer;

    uint32_t cb0_addr;
    cb_reserve_back(cb0_id, 1);
    cb0_addr = get_write_ptr(cb0_id);

    // One packet per issue; anything larger is split by hardware anyway.
    uint32_t pkt = transfer_size > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : transfer_size;
    uint32_t pkts_per_transfer = transfer_size / pkt;
    if (pkts_per_transfer < 1) {
        pkts_per_transfer = 1;
    }

    uint64_t dst_base = get_noc_addr(adjacent_noc_x, adjacent_noc_y, adjacent_noc_cb1);
    noc_async_write_one_packet_set_state(dst_base, pkt);

    for (uint32_t i = 0; i < iter_count; i++) {
        for (uint32_t j = 0; j < pkts_per_transfer; j++) {
            noc_async_write_one_packet_with_state(cb0_addr + j * pkt, adjacent_noc_cb1 + j * pkt);
        }
    }
    noc_async_write_barrier();
}
