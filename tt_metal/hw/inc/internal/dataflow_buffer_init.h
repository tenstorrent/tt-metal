// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/dataflow_buffer_interface.h"
#include "internal/circular_buffer_interface.h"  // for cb_addr_shift
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#include "internal/tt-2xx/quasar/overlay/remapper_api.hpp"
#include "internal/tt-2xx/quasar/overlay/dataflow_buffer_isr.h"
#endif

#include "api/debug/dprint.h"

// Global DFB interface array - defined in firmware, declared here for use by setup functions
// For kernels (NCRISC/BRISC/TRISC), provide a definition since they're compiled separately
extern RemapperAPI g_remapper_configurator;

namespace experimental {

extern thread_local LocalDFBInterface g_dfb_interface[32];

FORCE_INLINE void setup_isr_csrs() {
    uint64_t csr_reg;
    uint64_t old_mtvec;
    asm volatile("csrr %0, mtvec" : "=r"(old_mtvec));

    // Setup mtvec
    csr_reg = (uint64_t)dfb_implicit_sync_handler;

    asm volatile("csrw mtvec, %0" : : "r"(csr_reg));

    // Enable ROCC interrupts in mie
    csr_reg = 1 << 13;
    asm volatile("csrw mie, %0" : : "r"(csr_reg));

    // Enable mie in mstatus
    uint64_t other_csr_reg;
    asm volatile("csrr %0, mstatus" : "=r"(other_csr_reg));
    other_csr_reg |= 1 << 3;
    asm volatile("csrw mstatus, %0" : : "r"(other_csr_reg));
}

FORCE_INLINE void setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t local_dfb_mask) {
    uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    uint16_t hart_bit = 1 << hartid;

    uint32_t num_dfbs =
        local_dfb_mask;  // kernel config holds local_cb_mask but it gets hijacked to hold number of dfbs
    volatile uint8_t* base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);

    bool enable_remapper = false;  // if remapper used once then needs to be globally set

    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        // Read dfb_initializer_t (shared config)
        volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
        // TODO: update risc mask handling for tensix
        uint16_t risc_mask = init_ptr->risc_mask_bits.dm_mask;
        uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

        // Per-risc configs start after dfb_initializer_t
        volatile dfb_initializer_per_risc_t* per_risc_base =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

        if (risc_mask & hart_bit) {
            // Find this risc's per-risc config by counting set bits before this position
            uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & ((1 << hartid) - 1)));
            volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

            // Populate LocalDFBInterface from combined dfb_initializer_t + dfb_initializer_per_risc_t
            LocalDFBInterface& dfb_interface = ::g_dfb_interface[logical_dfb_id];

            DPRINT << "risc_index: " << static_cast<uint32_t>(risc_index) << ENDL();

            // Copy per-risc fields
            dfb_interface.num_tcs_to_rr = per_risc_ptr->num_tcs_to_rr;
            uint8_t tile_counters[MAX_NUM_TILE_COUNTERS_TO_RR];
            DPRINT << "num_tcs_to_rr: " << static_cast<uint32_t>(dfb_interface.num_tcs_to_rr) << ENDL();
            for (uint8_t i = 0; i < per_risc_ptr->num_tcs_to_rr; i++) {
                dfb_interface.base_addr[i] = per_risc_ptr->base_addr[i] >> cb_addr_shift;
                DPRINT << "base_addr[" << static_cast<uint32_t>(i) << "]: " << dfb_interface.base_addr[i] << ENDL();
                dfb_interface.limit[i] = per_risc_ptr->limit[i] >> cb_addr_shift;
                DPRINT << "limit[" << static_cast<uint32_t>(i) << "]: " << dfb_interface.limit[i] << ENDL();
                dfb_interface.rd_ptr[i] = per_risc_ptr->base_addr[i] >> cb_addr_shift;
                DPRINT << "rd_ptr[" << static_cast<uint32_t>(i) << "]: " << dfb_interface.rd_ptr[i] << ENDL();
                dfb_interface.wr_ptr[i] = per_risc_ptr->base_addr[i] >> cb_addr_shift;
                DPRINT << "wr_ptr[" << static_cast<uint32_t>(i) << "]: " << dfb_interface.wr_ptr[i] << ENDL();
                dfb_interface.packed_tile_counter[i] = per_risc_ptr->packed_tile_counter[i];
                DPRINT << "packed_tile_counter[" << static_cast<uint32_t>(i)
                       << "]: " << (uint32_t)dfb_interface.packed_tile_counter[i] << ENDL();
                tile_counters[i] = per_risc_ptr->packed_tile_counter[i];
            }

            dfb_interface.entry_size = init_ptr->entry_size;
            DPRINT << "entry_size: " << static_cast<uint32_t>(dfb_interface.entry_size) << ENDL();
            dfb_interface.stride_size = init_ptr->stride_size;
            DPRINT << "stride_size: " << static_cast<uint32_t>(dfb_interface.stride_size) << ENDL();

            dfb_interface.remapper_pair_index = per_risc_ptr->flags.remapper_pair_index;
            DPRINT << "remapper_pair_index: " << static_cast<uint32_t>(dfb_interface.remapper_pair_index) << ENDL();

#ifndef COMPILE_FOR_TRISC
            if (per_risc_ptr->init_txn_id_descriptor != 0) {
                dfb_interface.num_txn_ids = per_risc_ptr->num_txn_ids;
                // This descriptor is used by the ISR to understand which tile counters need to update which credits
                // (post/ack)
                TxnDFBDescriptor txn_dfb_descriptor{
                    .num_counters = per_risc_ptr->num_tcs_to_rr,
                    .tile_counters = tile_counters,
                };
                if (per_risc_ptr->flags.should_init_tc) {  // producer
                    txn_dfb_descriptor.tiles_to_post = init_ptr->num_entries_to_process_threshold_producer;
                } else {
                    txn_dfb_descriptor.tiles_to_ack = init_ptr->num_entries_to_process_threshold_consumer;
                }
                uint32_t txn_id_mask = 0;
                for (uint8_t i = 0; i < per_risc_ptr->num_txn_ids; i++) {
                    uint8_t txn_id = per_risc_ptr->txn_ids[i];
                    dfb_interface.txn_ids[i] = txn_id;
                    g_txn_dfb_descriptor[txn_id] = txn_dfb_descriptor;
                    if (per_risc_ptr->flags.should_init_tc) {
                        SET_TILES_TO_PROCESS_THRES_TR_ACK(txn_id, txn_dfb_descriptor.tiles_to_post);
                    } else {
                        SET_TILES_TO_PROCESS_THRES_WR_SENT(txn_id, txn_dfb_descriptor.tiles_to_ack);
                    }
                    txn_id_mask |= (1u << txn_id);
                }
                if (per_risc_ptr->flags.should_init_tc) {
                    per_trid_tiles_to_process_set_interrupt_enable_cmdbuf_0(txn_id_mask);
                } else {
                    per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf_0(txn_id_mask);
                }
                dfb_interface.num_entries_per_txn_id = per_risc_ptr->num_entries_per_txn_id;
                dfb_interface.num_entries_per_txn_id_per_tc = per_risc_ptr->num_entries_per_txn_id_per_tc;

                if (hartid == 0) {  // TODO: should specify only 1 risc does this but not necessarily DM0
                    setup_isr_csrs();
                }
            }

            // Configure remapper if needed (must be done before TC init)
            if (per_risc_ptr->flags.should_init_tc && per_risc_ptr->flags.remapper_en) {
                if (risc_index == 0) {  // update this
                    enable_remapper = true;
                }
                uint8_t remapper_consumer_mask = init_ptr->remapper_consumer_mask;
                uint8_t num_clientRs = __builtin_popcount(remapper_consumer_mask);
                uint8_t clientR_valid_mask = (1u << num_clientRs) - 1;
                g_remapper_configurator.set_pair_index(dfb_interface.remapper_pair_index);
                DPRINT << "Setting clientL fields " << static_cast<uint32_t>(risc_index)
                       << " id: " << static_cast<uint32_t>(get_counter_id(per_risc_ptr->packed_tile_counter[0]))
                       << " mask: " << static_cast<uint32_t>(clientR_valid_mask) << ENDL();
                g_remapper_configurator.configure_clientL_all_fields(
                    risc_index,                                            // id_L
                    get_counter_id(per_risc_ptr->packed_tile_counter[0]),  // in SxB mode, producers have 1 TC
                    clientR_valid_mask,
                    1,  // is_producer
                    1,  // group mode
                    0   // distribute mode
                );
                // explore each consumer programming their R slot rather than producer doing all
                for (uint8_t clientR_idx = 0; clientR_idx < num_clientRs; clientR_idx++) {
                    uint8_t mask = remapper_consumer_mask;
                    for (uint8_t i = 0; i < clientR_idx; i++) {
                        mask &= mask - 1;
                    }
                    uint8_t id_R = 4;  //__builtin_ctz(mask);
                    uint8_t tc_R =
                        (per_risc_ptr->consumer_tcs >> (clientR_idx * 5)) & 0x1F;  // TC can be value between 0 and 31
                    DPRINT << "Setting clientR slot " << static_cast<uint32_t>(clientR_idx)
                           << " id: " << static_cast<uint32_t>(id_R) << " tc: " << static_cast<uint32_t>(tc_R)
                           << ENDL();
                    g_remapper_configurator.set_clientR_slot(clientR_idx, id_R, tc_R);
                }
                DPRINT << "Writing all remapper configs" << ENDL();
                g_remapper_configurator.write_all_configs();
            }
#endif
        }

        // Jump to next DFB: skip dfb_initializer_t + (num_riscs * dfb_initializer_per_risc_t)
        base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
    }

    // all DFBs were initialized, safe to enable remapper if used
    if (enable_remapper && hartid == 0) {  // update how one risc enables the remapper
        DPRINT << "Enabling remapper" << ENDL();
        g_remapper_configurator.enable_remapper();
    }

#ifndef COMPILE_FOR_TRISC
    // Initialize TCs after remapper is enabled - only the RISC marked as responsible should do this (producer)
    base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);
    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
        uint16_t risc_mask = init_ptr->risc_mask_bits.dm_mask;
        uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

        volatile dfb_initializer_per_risc_t* per_risc_base =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

        if (risc_mask & hart_bit) {
            uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & ((1 << hartid) - 1)));
            volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

            if (per_risc_ptr->flags.should_init_tc) {
                for (uint8_t tc = 0; tc < per_risc_ptr->num_tcs_to_rr; tc++) {
                    PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                    uint8_t tensix_id = get_tensix_id(ptc);
                    uint8_t tc_id = get_counter_id(ptc);

                    DPRINT << "initializing tc tensix_id: " << static_cast<uint32_t>(tensix_id)
                           << " tc_id: " << static_cast<uint32_t>(tc_id) << ENDL();

                    llk_intf_reset(tensix_id, tc_id);
                    llk_intf_set_capacity(tensix_id, tc_id, init_ptr->capacity);
                }

                init_ptr->risc_mask_bits.tc_initialized = 1;
            }
        }

        base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
    }
#endif

    // After setting up g_dfb_interface, wait for all TCs to be initialized
    bool all_tcs_initialized = false;
    while (!all_tcs_initialized) {
        all_tcs_initialized = true;
        base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
            // TODO: update risc mask handling for tensix
            uint16_t risc_mask = init_ptr->risc_mask_bits.dm_mask;
            uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

            // TODO: Ring buffer is in uncached region so its okay to poll value. Needs to be uplifted when caching is
            // supported
            all_tcs_initialized &= init_ptr->risc_mask_bits.tc_initialized;

            base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
        }
    }
}

}  // namespace experimental
