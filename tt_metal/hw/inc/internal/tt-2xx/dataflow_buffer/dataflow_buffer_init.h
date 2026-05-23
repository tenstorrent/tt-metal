// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_config.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_interface.h"
#include "internal/tt-2xx/dataflow_buffer/dataflow_buffer_isr.h"
#include "internal/tt-2xx/quasar/overlay/remapper_api.hpp"
#include "internal/circular_buffer_interface.h"  // for cb_addr_shift
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#else
#include "ckernel_trisc_common.h"
#endif

#include "api/debug/dprint.h"


// Poll until every DFB has all producer TCs initialized and DM0's ISR path marked done.
// Item 2 (early-break): break as soon as any condition fails to avoid scanning remaining DFBs.
// Item 3 (move-implicit-sync): implicit_sync_configured is now written once by DM0 in
//   setup_local_dfb_interfaces; we only read it here.
FORCE_INLINE void wait_all_tcs_initialized(uint32_t tt_l1_ptr* dfb_config_base, uint32_t num_dfbs, uint64_t hartid) {
    WAYPOINT("TCIW");
    bool all_tcs_initialized = false;
    while (!all_tcs_initialized) {
        all_tcs_initialized = true;
        volatile uint8_t* base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);

            if (init_ptr->implicit_sync_configured != 1) {
                all_tcs_initialized = false;
                break;
            }

            uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
            uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

            volatile dfb_initializer_per_risc_t* per_risc_base =
                reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

            uint8_t producers_done = 0;
            for (uint8_t i = 0; i < num_riscs; i++) {
                if (per_risc_base[i].flags.is_producer && per_risc_base[i].num_tcs_and_init.tc_init_done) {
                    producers_done++;
                }
            }
            if (producers_done != init_ptr->num_producers) {
                all_tcs_initialized = false;
                break;
            }

            base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
        }
    }
    WAYPOINT("TCID");
}

inline uint32_t rdcycle() {
      uint32_t c;
      asm volatile("rdcycle %0" : "=r"(c));
      return c;
  }

// Item 1 (merge-passes):
//   DM path:   one merged loop (old Pass 1 + DM0's Pass 2 data-collection, in lock-step per DFB),
//              followed by DM0's post-loop (remapper enable + ISR setup + implicit_sync_configured write),
//              followed by a TC-init loop (old Pass 3) for all DMs.
//   TRISC path: one merged loop (old Pass 1 + old Pass 3) — no second loop needed.
FORCE_INLINE void setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t local_dfb_mask) {
    uint32_t start_time = rdcycle();

    uint64_t hartid;
#ifdef COMPILE_FOR_TRISC
    std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    // Building up g_dfb_interface is not at granularity of trisc in a Neo so only need Neo ID here
    // The initialization structs track producers/consumers for a given DFB and they would only be used by one of the
    // unpacker or packer
    hartid = 8 + neo_id;
#else
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
#endif
    uint16_t hart_bit = 1 << hartid;

    uint32_t num_dfbs =
        local_dfb_mask;  // kernel config holds local_cb_mask but it gets hijacked to hold number of dfbs
    volatile uint8_t* base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    uint8_t compact_dfb_count = 0;
#endif

#ifndef COMPILE_FOR_TRISC
    // DM0-only accumulator state for the post-loop remapper enable and ISR setup.
    bool enable_remapper = false;
    uint32_t producer_txn_id_mask = 0;
    uint32_t consumer_txn_id_mask = 0;
    uint32_t end_remapper_config_time = 0;
    uint32_t end_isr_enable_time = 0;
#endif

    // -----------------------------------------------------------------------
    // Merged loop: populate g_dfb_interface for this RISC (all RISCs),
    //              + collect TCs / configure remapper / set up TxnDFBDescriptors for DM0,
    //              + TC hardware init for TRISC producers (UCK_CHLKC_PACK only).
    // -----------------------------------------------------------------------
    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        // Read dfb_initializer_t (shared config)
        volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
        uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
        uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

        // Per-risc configs start after dfb_initializer_t
        volatile dfb_initializer_per_risc_t* per_risc_base =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

        // --- Sub-pass A: populate g_dfb_interface (all RISCs) ---
        if (risc_mask & hart_bit) {
            uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & ((1 << hartid) - 1)));
            volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
            ASSERT(compact_dfb_count < dfb::MAX_ACTIVE_DFBS_PACK);
            const uint8_t compact_dfb_id = compact_dfb_count++;
            g_dfb_logical_to_compact[logical_dfb_id] = compact_dfb_id;
            LocalDFBInterface& dfb_interface = g_dfb_interface[compact_dfb_id];
#else
            LocalDFBInterface& dfb_interface = g_dfb_interface[logical_dfb_id];
#endif

            dfb_interface.num_tcs_to_rr = per_risc_ptr->num_tcs_and_init.num_tcs_to_rr;

#ifdef COMPILE_FOR_TRISC
            dfb_interface.entry_size = static_cast<uint16_t>(init_ptr->entry_size >> cb_addr_shift);
            dfb_interface.stride_size = static_cast<uint16_t>(
                static_cast<uint32_t>(dfb_interface.entry_size) * static_cast<uint32_t>(init_ptr->stride_in_entries));
            dfb_interface.stride_size_tiles = static_cast<uint8_t>(init_ptr->stride_in_entries);
#if defined(UCK_CHLKC_PACK)
            dfb_interface.wr_entry_ptr = 0;
#else
            dfb_interface.tensix_trisc_mask = static_cast<uint8_t>(init_ptr->risc_mask_bits.tensix_trisc_mask);
#endif
#else
            dfb_interface.entry_size = init_ptr->entry_size >> cb_addr_shift;
            dfb_interface.stride_size = dfb_interface.entry_size * init_ptr->stride_in_entries;
#endif

            for (uint8_t i = 0; i < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; i++) {
                uint32_t base = per_risc_ptr->tc_addrs[i].base_addr >> cb_addr_shift;
                uint32_t limit_s = per_risc_ptr->tc_addrs[i].limit >> cb_addr_shift;
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
                dfb_interface.tc_slots[i].base_addr = base;
                dfb_interface.tc_slots[i].wr_offset = 0;
                dfb_interface.tc_slots[i].ring_size = static_cast<uint16_t>(limit_s - base);
                dfb_interface.tc_slots[i].packed_tile_counter = per_risc_ptr->packed_tile_counter[i];
                dfb_interface.tc_slots[i].base_entry_idx = static_cast<uint16_t>(
                    (base - dfb_interface.tc_slots[0].base_addr) / dfb_interface.entry_size);
                dfb_interface.tc_slots[i].wr_entry_idx = dfb_interface.tc_slots[i].base_entry_idx;
#elif defined(COMPILE_FOR_TRISC)
                dfb_interface.tc_slots[i].base_addr = base;
                dfb_interface.tc_slots[i].rd_offset = 0;
                dfb_interface.tc_slots[i].ring_size = static_cast<uint16_t>(limit_s - base);
                dfb_interface.tc_slots[i].packed_tile_counter = per_risc_ptr->packed_tile_counter[i];
                dfb_interface.tc_slots[i].base_entry_idx = static_cast<uint16_t>(
                    (base - dfb_interface.tc_slots[0].base_addr) / dfb_interface.entry_size);
                dfb_interface.tc_slots[i].rd_entry_idx = dfb_interface.tc_slots[i].base_entry_idx;
#else
                dfb_interface.tc_slots[i].base_addr = base;
                dfb_interface.tc_slots[i].limit = limit_s;
                dfb_interface.tc_slots[i].rd_ptr = base;
                dfb_interface.tc_slots[i].wr_ptr = base;
                dfb_interface.tc_slots[i].packed_tile_counter = per_risc_ptr->packed_tile_counter[i];
#endif
            }

            dfb_interface.tc_idx = 0;
#ifndef COMPILE_FOR_TRISC
            dfb_interface.broadcast_tc = per_risc_ptr->num_tcs_and_init.broadcast_tc;

            if (per_risc_ptr->flags.is_producer) {
                dfb_interface.num_txn_ids = init_ptr->producer_txn_descriptor.num_txn_ids;
                dfb_interface.threshold = init_ptr->producer_txn_descriptor.num_entries_to_process_threshold;
                dfb_interface.num_entries_per_txn_id = init_ptr->producer_txn_descriptor.num_entries_per_txn_id;
                dfb_interface.num_entries_per_txn_id_per_tc =
                    init_ptr->producer_txn_descriptor.num_entries_per_txn_id_per_tc;
                for (uint8_t i = 0; i < dfb_interface.num_txn_ids; i++) {
                    dfb_interface.txn_ids[i] = init_ptr->producer_txn_descriptor.txn_ids[i];
                }
            } else {
                dfb_interface.num_txn_ids = init_ptr->consumer_txn_descriptor.num_txn_ids;
                dfb_interface.threshold = init_ptr->consumer_txn_descriptor.num_entries_to_process_threshold;
                dfb_interface.num_entries_per_txn_id = init_ptr->consumer_txn_descriptor.num_entries_per_txn_id;
                dfb_interface.num_entries_per_txn_id_per_tc =
                    init_ptr->consumer_txn_descriptor.num_entries_per_txn_id_per_tc;
                for (uint8_t i = 0; i < dfb_interface.num_txn_ids; i++) {
                    dfb_interface.txn_ids[i] = init_ptr->consumer_txn_descriptor.txn_ids[i];
                }
            }
#endif

            // --- Sub-pass B (TRISC UCK_CHLKC_PACK only): TC hardware init merged into this loop ---
            // All producers (DM and TRISC) must wait for DM0's remapper enable before initializing TCs.
            // TRISC producers can spin-wait here safely because DM0's remapper enable happens in
            // the post-loop block below, which DM0 reaches concurrently via the same merged loop.
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
            if (per_risc_ptr->flags.is_producer) {
                while (per_risc_ptr->flags.remapper_en && !overlay::RemapperAPI::is_remapper_enabled());
                for (uint8_t tc = 0; tc < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; tc++) {
                    dfb::PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                    uint8_t tc_id = dfb::get_counter_id(ptc);
                    ckernel::trisc::tile_counters[tc_id].f.reset = 1;
                    ckernel::trisc::tile_counters[tc_id].f.buf_capacity = init_ptr->capacity;
                }
                per_risc_ptr->num_tcs_and_init.tc_init_done = 1;
            }
#endif
        }  // end if (risc_mask & hart_bit)

#ifndef COMPILE_FOR_TRISC
        // --- Sub-pass B (DM0 only): collect TC sets, configure remapper, set up TxnDFBDescriptors ---
        // Merged from old Pass 2: avoids re-reading init_ptr / risc_mask / per_risc_base for DM0.
        if (hartid == 0) {
            uint8_t num_producer_tcs = 0;
            uint8_t producer_tcs[16] = {};
            uint8_t num_consumer_tcs = 0;
            uint8_t consumer_tcs[16] = {};

            for (uint8_t i = 0; i < num_riscs; i++) {
                volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + i;

                if (per_risc_ptr->flags.is_producer) {
                    if (per_risc_ptr->flags.remapper_en) {
                        enable_remapper = true;
                        uint8_t remapper_consumer_ids_mask = per_risc_ptr->remapper_consumer_ids_mask;
                        uint8_t producer_client_type = per_risc_ptr->producer_client_type;
                        uint8_t num_clientRs = static_cast<uint8_t>(__builtin_popcount(remapper_consumer_ids_mask));
                        uint8_t clientR_valid_mask = (1u << num_clientRs) - 1;
                        g_remapper_configurator.set_pair_index(
                            static_cast<uint32_t>(per_risc_ptr->flags.remapper_pair_index));
                        g_remapper_configurator.configure_clientL_all_fields(
                            producer_client_type,
                            dfb::get_counter_id(per_risc_ptr->packed_tile_counter[0]),
                            clientR_valid_mask,
                            1,  // is_producer
                            1,  // group mode
                            0   // distribute mode
                        );
                        uint8_t mask_remaining = remapper_consumer_ids_mask;
                        for (uint8_t clientR_idx = 0; clientR_idx < num_clientRs; clientR_idx++) {
                            uint8_t id_R = static_cast<uint8_t>(__builtin_ctz(mask_remaining));
                            mask_remaining &= mask_remaining - 1;
                            uint8_t tc_R = (per_risc_ptr->consumer_tcs >> (clientR_idx * 5)) & 0x1F;
                            g_remapper_configurator.set_clientR_slot(clientR_idx, id_R, tc_R);
                        }
                        g_remapper_configurator.write_all_configs();
                    }
                    for (uint8_t j = 0; j < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; j++) {
                        producer_tcs[num_producer_tcs++] = per_risc_ptr->packed_tile_counter[j];
                    }
                } else {
                    for (uint8_t j = 0; j < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; j++) {
                        consumer_tcs[num_consumer_tcs++] = per_risc_ptr->packed_tile_counter[j];
                    }
                }
            }

            for (uint8_t i = 0; i < init_ptr->producer_txn_descriptor.num_txn_ids; i++) {
                uint8_t txn_id = init_ptr->producer_txn_descriptor.txn_ids[i];
                producer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = static_cast<uint8_t>(num_producer_tcs);
                for (uint8_t j = 0; j < dst.num_counters; j++) {
                    dst.tile_counters[j] = producer_tcs[j];
                }
                dst.tiles_to_post = init_ptr->producer_txn_descriptor.num_entries_per_txn_id_per_tc;
                CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, txn_id);
                asm volatile("nop");
                SET_TILES_TO_PROCESS_THRES_TR_ACK(
                    txn_id, init_ptr->producer_txn_descriptor.num_entries_to_process_threshold);
            }
            for (uint8_t i = 0; i < init_ptr->consumer_txn_descriptor.num_txn_ids; i++) {
                uint8_t txn_id = init_ptr->consumer_txn_descriptor.txn_ids[i];
                consumer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = static_cast<uint8_t>(num_consumer_tcs);
                for (uint8_t j = 0; j < dst.num_counters; j++) {
                    dst.tile_counters[j] = consumer_tcs[j];
                }
                CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, txn_id);
                asm volatile("nop");
                dst.tiles_to_ack = init_ptr->consumer_txn_descriptor.num_entries_per_txn_id_per_tc;
                SET_TILES_TO_PROCESS_THRES_WR_SENT(
                    txn_id, init_ptr->consumer_txn_descriptor.num_entries_to_process_threshold);
            }
        }  // end DM0 sub-pass B
#endif

        base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
    }
    // -----------------------------------------------------------------------
    // End merged loop
    // -----------------------------------------------------------------------

#ifndef COMPILE_FOR_TRISC
    // DM0 post-loop: enable remapper, set up ISR interrupt enables, then write
    // implicit_sync_configured = 1 for all DFBs so other DMs/TRISCs can proceed.
    if (hartid == 0) {
        // Program which transaction ids should trigger the implicit sync ISR
        uint64_t reg_val = CMDBUF_RD_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(producer_txn_id_mask & 0xFFFFFFFFULL) << 32);
        CMDBUF_WR_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);

        reg_val = CMDBUF_RD_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (consumer_txn_id_mask & 0xFFFFFFFFULL);
        CMDBUF_WR_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);

        if (enable_remapper) {
            g_remapper_configurator.enable_remapper();
            end_remapper_config_time = rdcycle();
        }

        if ((producer_txn_id_mask | consumer_txn_id_mask) != 0) {
            enable_dfb_tile_isr();
            end_isr_enable_time = rdcycle();
        } else {
            disable_dfb_tile_isr();
        }
    }  // end if (hartid == 0) post-loop

    // -----------------------------------------------------------------------
    // TC init loop (old Pass 3): all DMs; DM producers spin-wait for remapper.
    // Item 3 (move-implicit-sync): DM0 writes implicit_sync_configured = 1 here,
    //   once per DFB, after ISR setup is complete — replacing the repeated volatile
    //   store that was inside every iteration of the wait_all_tcs_initialized poll loop.
    // -----------------------------------------------------------------------
    base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);
    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
        uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
        uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

        volatile dfb_initializer_per_risc_t* per_risc_base =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

        // DM0 signals ISR-ready to all other DMs and TRISCs once per DFB, not on every poll iteration.
        if (hartid == 0) {
            init_ptr->implicit_sync_configured = 1;
        }

        if (risc_mask & hart_bit) {
            uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & ((1 << hartid) - 1)));
            volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

            if (per_risc_ptr->flags.is_producer) {
                while (per_risc_ptr->flags.remapper_en && !overlay::RemapperAPI::is_remapper_enabled());
                for (uint8_t tc = 0; tc < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; tc++) {
                    dfb::PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                    uint8_t tensix_id = dfb::get_tensix_id(ptc);
                    uint8_t tc_id = dfb::get_counter_id(ptc);
                    overlay::llk_intf_reset(tensix_id, tc_id);
                    overlay::llk_intf_set_capacity(tensix_id, tc_id, init_ptr->capacity);
                }
                per_risc_ptr->num_tcs_and_init.tc_init_done = 1;
            }
        }

        base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
    }
#endif

    wait_all_tcs_initialized(dfb_config_base, num_dfbs, hartid);
    uint32_t end_time = rdcycle();

    // All DPRINTs deferred to here so zero DPRINT overhead falls inside the timed region.
    DPRINT << "start_time: " << start_time << ENDL();
#ifndef COMPILE_FOR_TRISC
    if (hartid == 0) {
        if (end_remapper_config_time) DPRINT << "end_remapper_config_time: " << end_remapper_config_time << ENDL();
        if (end_isr_enable_time) DPRINT << "end_isr_enable_time: " << end_isr_enable_time << ENDL();
    }
#endif
    DPRINT << "end_time: " << end_time << ENDL();
}
