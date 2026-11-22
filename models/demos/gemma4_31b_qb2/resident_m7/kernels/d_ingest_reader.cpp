// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter ingest, streamer reader (NCRISC; M6b C1), see d_ingest_common.hpp: d_streamer_reader.cpp's
// weight stream (own bank, pages alternating NoC 0 / 1, trid, <= 2 blocks in flight per NoC) over W_fc, then W_kv of
// every drafter layer. At the end one bump of the hub's record semaphore (hub_dm.cpp counts two per streamer; the
// ingest has no logits, the hub's argmax over the untouched records is never read).
//
// Runtime args: 0 bank_id, 1 vc, 2 slot j of this core in its bank, 3 cores_per_bank, then per entry e: 4 + 4e
// n_tiles, base address, per-copy stride (bytes per bank), copies; 12 hub_x, 13 hub_y, 14 hub sem_rec id

#include "api/dataflow/dataflow_api.h"
#include "d_ingest_common.hpp"

using namespace ding;

namespace {

constexpr uint32_t kInFlight = 2;

uint32_t ring_base;
volatile uint32_t* consumed;
RingCursor cursor;

FORCE_INLINE void wait_for_room() {
    constexpr uint32_t ring_units = ring_bytes / kUnit;
    while (((cursor.units - *consumed) & 0xffff) > ring_units) {
    }
}

// copy k (of this entry's kind) of the weights
template <uint32_t E>
FORCE_INLINE void stream_entry(uint32_t bank_id, uint32_t vc, uint32_t slot, uint32_t slots, uint32_t k) {
    using En = Entry<E>;
    const uint32_t n_tiles = get_arg_val<uint32_t>(4 + 4 * E);
    const uint32_t copies = get_arg_val<uint32_t>(7 + 4 * E);
    const uint32_t weight_addr = get_arg_val<uint32_t>(5 + 4 * E) + (k % copies) * get_arg_val<uint32_t>(6 + 4 * E);
    const uint32_t num_blocks = n_tiles * En::nkb;
    const uint64_t base0 = get_noc_addr_from_bank_id<true>(bank_id, weight_addr, 0);
    const uint64_t base1 = get_noc_addr_from_bank_id<true>(bank_id, weight_addr, 1);
    noc_async_read_one_packet_set_state<true>(base0, En::page_size, vc, 0);
    noc_async_read_one_packet_set_state<true>(base1, En::page_size, vc, 1);

    const uint32_t stride = slots * En::block_bytes;
    uint32_t src = slot * En::block_bytes;
    uint32_t issued = 0;
    uint32_t next_trid = 1;
    uint32_t wait_trid = 1;
    auto retire = [&]() {
        noc_async_read_barrier_with_trid(wait_trid, 0);
        noc_async_read_barrier_with_trid(wait_trid, 1);
        cb_push_back(En::cb, En::sb);
        wait_trid = wait_trid == kInFlight + 1 ? 1 : wait_trid + 1;
        issued--;
    };
    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t l1 = ring_base + cursor.place<E>();
        wait_for_room();
        noc_async_read_set_trid(next_trid, 0);
        noc_async_read_set_trid(next_trid, 1);
        for (uint32_t p = 0; p < En::pages; p++) {
            const uint32_t off = p * En::page_size;
            const uint8_t noc = (b * En::pages + p) & 1;
            noc_async_read_one_packet_with_state_with_trid(noc ? base1 : base0, src + off, l1 + off, next_trid, noc);
        }
        src += stride;
        next_trid = next_trid == kInFlight + 1 ? 1 : next_trid + 1;
        if (++issued == kInFlight + 1) {
            retire();
        }
    }
    while (issued > 0) {
        retire();
    }
}

}  // namespace

void kernel_main() {
    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t slot = get_arg_val<uint32_t>(2);
    const uint32_t slots = get_arg_val<uint32_t>(3);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 0);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, 1);
    {
        auto& w = get_local_cb_interface(cb_w0);
        ring_base = (w.fifo_limit - w.fifo_size) << cb_addr_shift;
        consumed = get_cb_tiles_acked_ptr(cb_consumed);
        *consumed = 0;
    }
    stream_entry<kFc>(bank_id, vc, slot, slots, 0);
    for (uint32_t l = 0; l < layers; l++) {
        stream_entry<kKv>(bank_id, vc, slot, slots, l);
    }
    const uint32_t hub_x = get_arg_val<uint32_t>(12), hub_y = get_arg_val<uint32_t>(13);
    noc_semaphore_inc(get_noc_addr(hub_x, hub_y, get_semaphore(get_arg_val<uint32_t>(14)), 0), 1, 0);
    noc_async_atomic_barrier(0);
}
