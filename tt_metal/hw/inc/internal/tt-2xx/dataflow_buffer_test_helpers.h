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
//
// This file lives under hw/inc/internal/tt-2xx/ (not under hw/inc/api/) so it
// does not advertise itself as part of the public DFB surface. Production
// kernels must NOT include this header.

#pragma once

#if defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)

#include <cstdint>

#include "api/dataflow/dataflow_buffer.h"

inline void preload_posted_counter(DataflowBuffer& dfb, uint16_t value) {
    for (uint8_t i = 0; i < dfb.local_dfb_interface_.num_tcs_to_rr; i++) {
        dfb::PackedTileCounter ptc = dfb.local_dfb_interface_.tc_slots[i].packed_tile_counter;
        overlay::llk_intf_inc_posted(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc), value);
    }
    const uint16_t per_tc = dfb.local_dfb_interface_.num_entries_per_txn_id_per_tc;
    if (per_tc > 0 && dfb.local_dfb_interface_.num_txn_ids > 0) {
        dfb.ptxn_id_loop_cnt_ = value / per_tc;
        dfb.ptxn_id_index_ = static_cast<uint8_t>(dfb.ptxn_id_loop_cnt_ % dfb.local_dfb_interface_.num_txn_ids);
    }
    // Bump the kernel-side transactions-issued counter so handle_final_credits's
    // (transactions_issued % N, / N) math matches the HW posted value at finish time.
    dfb.ptiles_read_ = value;
}

inline void preload_acked_counter(DataflowBuffer& dfb, uint16_t value) {
    for (uint8_t i = 0; i < dfb.local_dfb_interface_.num_tcs_to_rr; i++) {
        dfb::PackedTileCounter ptc = dfb.local_dfb_interface_.tc_slots[i].packed_tile_counter;
        overlay::llk_intf_inc_acked(dfb::get_tensix_id(ptc), dfb::get_counter_id(ptc), value);
    }
    const uint16_t per_tc = dfb.local_dfb_interface_.num_entries_per_txn_id_per_tc;
    if (per_tc > 0 && dfb.local_dfb_interface_.num_txn_ids > 0) {
        dfb.ctxn_id_loop_cnt_ = value / per_tc;
        dfb.ctxn_id_index_ = static_cast<uint8_t>(dfb.ctxn_id_loop_cnt_ % dfb.local_dfb_interface_.num_txn_ids);
    }
    dfb.ctiles_written_ = value;
}

#endif  // defined(ARCH_QUASAR) && !defined(COMPILE_FOR_TRISC)
