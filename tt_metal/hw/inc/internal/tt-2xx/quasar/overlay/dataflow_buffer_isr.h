// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/dataflow_buffer_interface.h"
// #include "internal/tt-2xx/quasar/overlay/cmdbuff_api.hpp"
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"

namespace experimental {

extern volatile TxnDFBDescriptor g_txn_dfb_descriptor[16];

inline __attribute__((always_inline)) void dfb_tile_poster_irq_handler() {
    // uint64_t pending_tr_tiles_to_process = per_trid_tiles_to_process_interrupts_pending_cmdbuf_0();
    uint64_t val = CMDBUF_RD_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET);
    uint64_t pending_tr_tiles_to_process = (val >> 32) & 0xFFFFFFFFULL;
    if (pending_tr_tiles_to_process) {
        uint64_t pending_first_lsb_trid = __builtin_ctzll(pending_tr_tiles_to_process);

        volatile TxnDFBDescriptor& txn_dfb_descriptor = g_txn_dfb_descriptor[pending_first_lsb_trid];

        for (uint8_t i = 0; i < txn_dfb_descriptor.num_counters; i++) {
            PackedTileCounter packed_tile_counter = txn_dfb_descriptor.tile_counters[i];
            uint8_t tensix_id = get_tensix_id(packed_tile_counter);
            uint8_t tc_id = get_counter_id(packed_tile_counter);
            fast_llk_intf_inc_posted(tensix_id, tc_id, txn_dfb_descriptor.tiles_to_post);
        }

        CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, pending_first_lsb_trid);
        asm volatile("nop");  // must give time for the clear to propagate (this is needed)

        // per_trid_tiles_to_process_interrupt_clear_cmdbuf_0(pending_first_lsb_trid);
        uint64_t clear_val = ~(1ULL << (pending_first_lsb_trid + 32));
        CMDBUF_WR_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET, clear_val);
    }
}

inline __attribute__((always_inline)) void dfb_tile_acker_irq_handler() {
    // uint64_t pending_tr_tiles_to_ack = per_trid_wr_tiles_to_process_interrupts_pending_cmdbuf_0();
    uint64_t val = CMDBUF_RD_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET);
    uint64_t pending_tr_tiles_to_ack = val & 0xFFFFFFFFULL;
    if (pending_tr_tiles_to_ack != 0) {
        uint64_t pending_first_lsb_trid = __builtin_ctzll(pending_tr_tiles_to_ack);

        volatile TxnDFBDescriptor& txn_dfb_descriptor = g_txn_dfb_descriptor[pending_first_lsb_trid];

        for (uint8_t i = 0; i < txn_dfb_descriptor.num_counters; i++) {
            PackedTileCounter packed_tile_counter = txn_dfb_descriptor.tile_counters[i];
            uint8_t tensix_id = get_tensix_id(packed_tile_counter);
            uint8_t tc_id = get_counter_id(packed_tile_counter);
            fast_llk_intf_inc_acked(tensix_id, tc_id, txn_dfb_descriptor.tiles_to_ack);
        }

        CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, pending_first_lsb_trid);
        asm volatile("nop");  // must give time for the clear to propagate (this is needed)

        // per_trid_wr_tiles_to_process_interrupt_clear_cmdbuf_0(pending_first_lsb_trid);
        uint64_t clear_val = ~(1ULL << pending_first_lsb_trid);
        CMDBUF_WR_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET, clear_val);                    \
    }
}

__attribute__((interrupt, hot)) void dfb_implicit_sync_handler() {
    dfb_tile_poster_irq_handler();
    dfb_tile_acker_irq_handler();
}

}  // namespace experimental