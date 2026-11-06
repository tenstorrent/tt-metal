// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Test-only DFB helpers. NOT part of the public DFB API.
//
// preload_*_counter exists solely so the D1 implicit-sync uint16-wrap test can
// fast-forward DFB credit counters into a near-wrap state BEFORE the kernel
// issues any reserve_back / push_back / async_read / async_write. Both the HW
// counter (posted/acked) AND the kernel-side shadow state (loop_cnt + txn_id
// index + tiles_*) must be advanced together so they stay in the same epoch —
// otherwise prepare_implicit_*'s modular comparison can't detect "caught up."
//
// Usage rules:
//   - Call AFTER the DFB has been constructed (`DataflowBuffer dfb(...)`)
//   - Call BEFORE any reserve_back / push_back / async_read / async_write
//   - Producer side updates posted; consumer side updates acked
//   - Caller is responsible for matching the values on both sides so the ring
//     still looks empty (posted - acked == 0)
//   - To fast-forward without ever leaving the counter out of range (acked ahead
//     of posted, or more posted than the ring holds), let ONE hart raise both
//     hardware counters together with preload_counters_lockstep and have each
//     side set only its software cursor with preload_*_cursor
//
// This file lives under hw/inc/internal/tt-2xx/ (not under hw/inc/api/) so it
// does not advertise itself as part of the public DFB surface. Production
// kernels must NOT include this header.

#pragma once

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void preload_posted_cursor(DataflowBuffer<Pap, Cap>& dfb, uint16_t value) {
    ASSERT(value % dfb.get_producer_share() == 0);
    const uint16_t per_tc = dfb.local_dfb_interface_.num_entries_per_txn_id_per_tc;
    if (per_tc > 0 && dfb.local_dfb_interface_.num_txn_ids > 0) {
        dfb.ptxn_id_loop_cnt_ = value / per_tc;
        dfb.ptxn_id_index_ = static_cast<uint8_t>(dfb.ptxn_id_loop_cnt_ % dfb.local_dfb_interface_.num_txn_ids);
    }
    // `value` was added to every round-robin counter. The software cursor counts
    // issues globally, so it is value * N (broadcast posts one issue to every
    // counter, so the cursor stays `value`). N is 1 for the D1 wrap test.
    dfb.ptiles_read_ = dfb.local_dfb_interface_.broadcast_tc
                           ? value
                           : static_cast<uint32_t>(value) * dfb.local_dfb_interface_.num_tcs_to_rr;
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void preload_acked_cursor(DataflowBuffer<Pap, Cap>& dfb, uint16_t value) {
    ASSERT(value % dfb.get_consumer_share() == 0);
    const uint16_t per_tc = dfb.local_dfb_interface_.num_entries_per_txn_id_per_tc;
    if (per_tc > 0 && dfb.local_dfb_interface_.num_txn_ids > 0) {
        dfb.ctxn_id_loop_cnt_ = value / per_tc;
        dfb.ctxn_id_index_ = static_cast<uint8_t>(dfb.ctxn_id_loop_cnt_ % dfb.local_dfb_interface_.num_txn_ids);
    }
    dfb.ctiles_written_ = dfb.local_dfb_interface_.broadcast_tc
                              ? value
                              : static_cast<uint32_t>(value) * dfb.local_dfb_interface_.num_tcs_to_rr;
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void preload_counters_lockstep(DataflowBuffer<Pap, Cap>& dfb, uint16_t value) {
    for (uint8_t i = 0; i < dfb.local_dfb_interface_.num_tcs_to_rr; i++) {
        const dfb::PackedTileCounter ptc = dfb.local_dfb_interface_.tc_slots[i].packed_tile_counter;
        const auto tid = dfb::get_tensix_id(ptc);
        const auto cid = dfb::get_counter_id(ptc);
        const uint16_t max_step = static_cast<uint16_t>(overlay::fast_llk_intf_get_capacity(tid, cid));
        ASSERT(max_step > 0);
        uint16_t remaining = value;
        while (remaining > 0) {
            const uint16_t step = remaining < max_step ? remaining : max_step;
            overlay::llk_intf_inc_posted(tid, cid, step);
            overlay::llk_intf_inc_acked(tid, cid, step);
            remaining -= step;
        }
    }
    // Let the counter writes land before the caller publishes anything that depends on them.
    asm volatile("fence w, w" ::: "memory");
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void preload_posted_counter(DataflowBuffer<Pap, Cap>& dfb, uint16_t value) {
    for (uint8_t i = 0; i < dfb.local_dfb_interface_.num_tcs_to_rr; i++) {
        dfb::PackedTileCounter ptc = dfb.local_dfb_interface_.tc_slots[i].packed_tile_counter;
        overlay::llk_intf_inc_posted(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc), value);
    }
    preload_posted_cursor(dfb, value);
}

template <dfb::AccessPattern Pap, dfb::AccessPattern Cap>
inline void preload_acked_counter(DataflowBuffer<Pap, Cap>& dfb, uint16_t value) {
    for (uint8_t i = 0; i < dfb.local_dfb_interface_.num_tcs_to_rr; i++) {
        dfb::PackedTileCounter ptc = dfb.local_dfb_interface_.tc_slots[i].packed_tile_counter;
        overlay::llk_intf_inc_acked(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc), value);
    }
    preload_acked_cursor(dfb, value);
}

#endif  // defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
