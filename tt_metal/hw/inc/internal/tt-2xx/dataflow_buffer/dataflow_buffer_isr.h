// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#endif

inline __attribute__((always_inline)) void enable_dfb_tile_isr();
inline __attribute__((always_inline)) void disable_dfb_tile_isr();

inline __attribute__((always_inline)) void dfb_tile_poster_irq_handler() {
#ifndef COMPILE_FOR_TRISC
    uint64_t fired_trids = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8);
    uint64_t pending = (fired_trids >> 32) & 0xFFFFFFFFULL;
    while (pending) {
        uint64_t trid = __builtin_ctzll(pending);

        volatile TxnDFBDescriptor& txn_dfb_descriptor = g_txn_dfb_descriptor[trid];
        for (uint8_t i = 0; i < txn_dfb_descriptor.num_counters; i++) {
            dfb::PackedTileCounter packed_tile_counter = txn_dfb_descriptor.tile_counters[i];
            uint8_t tensix_id = dfb::get_tensix_id(packed_tile_counter);
            uint8_t tc_id = dfb::get_counter_id(packed_tile_counter);
            overlay::fast_llk_intf_inc_posted(tensix_id, tc_id, txn_dfb_descriptor.tiles_to_post);
        }

        __builtin_riscv_ttrocc_cmdbuf_clear_tiles_to_process_tr_ack(OVERLAY_RD_CMD_BUF, trid);
        asm volatile("nop");  // must give time for the clear to propagate (hw bug, this is needed)

        pending &= (pending - 1);
    }
    if ((fired_trids >> 32) != 0) {
        // W0C: write 0 to clear a bit, write 1 to leave it unchanged
        uint64_t to_clear = (fired_trids >> 32) << 32;
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_1_REG_OFFSET / 8, ~to_clear);
    }
#endif
}

inline __attribute__((always_inline)) void dfb_tile_acker_irq_handler() {
#ifndef COMPILE_FOR_TRISC
    uint64_t fired_trids = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8);
    uint64_t pending = fired_trids & 0xFFFFFFFFULL;
    while (pending) {
        uint64_t trid = __builtin_ctzll(pending);

        volatile TxnDFBDescriptor& txn_dfb_descriptor = g_txn_dfb_descriptor[trid];
        for (uint8_t i = 0; i < txn_dfb_descriptor.num_counters; i++) {
            dfb::PackedTileCounter packed_tile_counter = txn_dfb_descriptor.tile_counters[i];
            uint8_t tensix_id = dfb::get_tensix_id(packed_tile_counter);
            uint8_t tc_id = dfb::get_counter_id(packed_tile_counter);
            overlay::fast_llk_intf_inc_acked(tensix_id, tc_id, txn_dfb_descriptor.tiles_to_ack);
        }

        __builtin_riscv_ttrocc_cmdbuf_clear_tiles_to_process_wr_sent(OVERLAY_WR_CMD_BUF, trid);
        asm volatile("nop");  // must give time for the clear to propagate (this is needed)

        pending &= (pending - 1);
    }
    if ((fired_trids & 0xFFFFFFFFULL) != 0) {
        // W0C: write 0 to clear a bit, write 1 to leave it unchanged
        uint64_t to_clear = fired_trids & 0xFFFFFFFFULL;
        __builtin_riscv_ttrocc_cmdbuf_wr_reg(
            OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IP_2_REG_OFFSET / 8, ~to_clear);
    }
#endif
}

inline __attribute__((interrupt, hot)) void dfb_implicit_sync_handler() {
#ifndef COMPILE_FOR_TRISC
    dfb_tile_poster_irq_handler();
    dfb_tile_acker_irq_handler();
#endif
}
