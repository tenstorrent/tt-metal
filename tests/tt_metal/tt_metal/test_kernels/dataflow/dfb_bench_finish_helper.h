// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark-only producer finish for single-producer implicit-sync DFBs.
//
// DataflowBuffer::finish() -> handle_final_credits() unconditionally calls
// sync_threads(get_num_threads()). Quasar benchmark kernels are often launched
// with num_threads_per_cluster > 1 while only one DM produces on a given DFB,
// which deadlocks in that barrier.
//
// Call dfb_finish_single_implicit_read_producer() after exactly one
// dfb_issue_implicit_read() on the same DFB handle. Skips the collective
// sync_threads barrier (not needed when a single DM owns the producer side)
// but otherwise mirrors finish_impl() for one implicit-sync read.

#pragma once

#include "api/dataflow/dataflow_buffer.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#include "internal/tt-2xx/quasar/overlay/rocc_instructions.hpp"

FORCE_INLINE void dfb_finish_single_implicit_read_producer(DataflowBuffer& dfb) {
    LocalDFBInterface& iface = get_local_dfb_interface(dfb.get_id());

    // State after exactly one commit_implicit_read() from a fresh DataflowBuffer.
    constexpr uint16_t transactions_issued = 1;
    const uint8_t txn_id_index = (transactions_issued % iface.num_entries_per_txn_id == 0)
                                       ? static_cast<uint8_t>((0 + 1) % iface.num_txn_ids)
                                       : 0;

    const uint8_t tail_txn_idx = (transactions_issued % iface.num_entries_per_txn_id == 0)
                                     ? static_cast<uint8_t>((txn_id_index + iface.num_txn_ids - 1) % iface.num_txn_ids)
                                     : txn_id_index;
    const uint8_t tail_txn_id = iface.txn_ids[tail_txn_idx];

    const uint8_t N = iface.num_tcs_to_rr;
    const dfb::PackedTileCounter ptc0 = iface.tc_slots[0].packed_tile_counter;
    const uint16_t expected_slot0 = transactions_issued / N + (0u < (transactions_issued % N) ? 1u : 0u);

    auto read_actual_slot0 = [&]() -> uint16_t {
        return static_cast<uint16_t>(
            overlay::fast_llk_intf_read_posted(dfb::get_tensix_id(ptc0), dfb::get_counter_id(ptc0)));
    };

    while (read_actual_slot0() < expected_slot0) {
        const uint64_t tack  = CMDBUF_TR_ACK_TRID(OVERLAY_RD_CMD_BUF, tail_txn_id);
        const uint64_t tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
        if (tack == 0 && tiles > 0) {
            break;
        }
    }

    if (read_actual_slot0() >= expected_slot0) {
        // ISR posted credits; fall through to drain wait below.
    } else {
        const uint16_t global_threshold = iface.threshold;
        while (read_actual_slot0() < expected_slot0) {
            const uint64_t tiles = CMDBUF_READ_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, tail_txn_id);
            if (tiles > 0 && tiles < global_threshold) {
                break;
            }
        }

        const uint16_t actual_slot0 = read_actual_slot0();
        if (actual_slot0 < expected_slot0) {
            for (uint8_t i = 0; i < N; i++) {
                const dfb::PackedTileCounter ptc = iface.tc_slots[i].packed_tile_counter;
                const uint8_t tensix_id = dfb::get_tensix_id(ptc);
                const uint8_t tc_id     = dfb::get_counter_id(ptc);
                const uint16_t expected = transactions_issued / N + (i < (transactions_issued % N) ? 1u : 0u);
                const uint16_t actual = static_cast<uint16_t>(overlay::fast_llk_intf_read_posted(tensix_id, tc_id));
                if (actual < expected) {
                    overlay::fast_llk_intf_inc_posted(tensix_id, tc_id, expected - actual);
                }
            }
        }
    }

    bool all_acked = false;
    while (!all_acked) {
        all_acked = true;
        for (uint8_t i = 0; i < iface.num_tcs_to_rr; i++) {
            const dfb::PackedTileCounter packed_tc = iface.tc_slots[i].packed_tile_counter;
            const uint8_t tc_id = dfb::get_counter_id(packed_tc);
            const uint8_t tensix_id = dfb::get_tensix_id(packed_tc);
            const uint32_t read_posted = overlay::fast_llk_intf_read_posted(tensix_id, tc_id);
            const uint32_t read_acked = overlay::fast_llk_intf_read_acked(tensix_id, tc_id);
            if (read_acked != read_posted) {
                all_acked = false;
            }
        }
    }
}
