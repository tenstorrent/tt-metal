// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/dataflow_buffer_interface.h"
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"

namespace experimental {

extern volatile TxnDFBDescriptor g_txn_dfb_descriptor[MAX_TOTAL_TXN_IDS];

__attribute__((interrupt, hot)) __attribute__((always_inline)) void dfb_tile_poster_irq_handler() {
    uint64_t pending_tr_tiles_to_process = per_trid_tiles_to_process_interrupts_pending_cmdbuf_0();
    uint64_t pending_first_lsb_trid = __builtin_ctzll(pending_tr_tiles_to_process);

    TxnDFBDescriptor& txn_dfb_descriptor = g_txn_dfb_descriptor[pending_first_lsb_trid];

    for (uint8_t i = 0; i < txn_dfb_descriptor.num_counters; i++) {
        PackedTileCounter packed_tile_counter = txn_dfb_descriptor.tile_counters[i];
        uint8_t tensix_id = get_tensix_id(packed_tc);
        uint8_t tc_id = get_counter_id(packed_tc);
        fast_llk_intf_inc_posted(tensix_id, tc_id, txn_dfb_descriptor.tiles_to_post);
    }

    CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(CMDBUF_0, pending_first_lsb_trid);
    asm volatile("nop");  // must give time for the clear to propagate (this is needed)

    per_trid_tiles_to_process_interrupt_clear_cmdbuf_0(pending_first_lsb_trid);
}

__attribute__((interrupt, hot)) __attribute__((always_inline)) void dfb_tile_acker_irq_handler() {
    uint64_t pending_tr_tiles_to_ack = per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf_0();
    if (pending_tr_tiles_to_ack != 0) {
        uint64_t pending_first_lsb_trid = __builtin_ctzll(pending_tr_tiles_to_ack);

        TxnDFBDescriptor& txn_dfb_descriptor = g_txn_dfb_descriptor[pending_first_lsb_trid];

        for (uint8_t i = 0; i < txn_dfb_descriptor.num_counters; i++) {
            PackedTileCounter packed_tile_counter = txn_dfb_descriptor.tile_counters[i];
            uint8_t tensix_id = get_tensix_id(packed_tc);
            uint8_t tc_id = get_counter_id(packed_tc);
            fast_llk_intf_inc_acked(tensix_id, tc_id, txn_dfb_descriptor.tiles_to_post);
        }

        CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(CMDBUF_0, pending_first_lsb_trid);
        asm volatile("nop");  // must give time for the clear to propagate (this is needed)

        per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf_0(pending_first_lsb_trid);
    }
}

__attribute__((interrupt, hot)) __attribute__((always_inline)) void dfb_implicit_sync_handler() {
    uint64_t pending_tr_tiles_to_post = per_trid_tiles_to_process_interrupts_pending_cmdbuf_0();
    uint64_t pending_tr_tiles_to_ack = per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf_0();

    if (pending_tr_tiles_to_post != 0) {
        dfb_tile_poster_irq_handler();
    }
    if (pending_tr_tiles_to_ack != 0) {
        dfb_tile_acker_irq_handler();
    }
}

}  // namespace experimental
