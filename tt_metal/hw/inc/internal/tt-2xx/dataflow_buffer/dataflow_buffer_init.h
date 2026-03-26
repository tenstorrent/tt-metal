// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
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

FORCE_INLINE void setup_isr_csrs() {
#ifndef COMPILE_FOR_TRISC
    // Setup mtvec
    uint64_t csr_reg = (uint64_t)dfb_implicit_sync_handler;

    asm volatile("csrw mtvec, %0" : : "r"(csr_reg));

    // Enable ROCC interrupt in mie
    asm volatile("csrrs zero, mie, %0" : : "r"(static_cast<uint64_t>(1 << 13)));

    // Enable MIE in mstatus
    asm volatile("csrrs zero, mstatus, %0" : : "r"(static_cast<uint64_t>(1 << 3)));
#endif
}

FORCE_INLINE void setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t local_dfb_mask) {
    uint64_t hartid;
#ifdef COMPILE_FOR_TRISC
    std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    // Building up g_dfb_interface is not at granularity of trisc in a Neo so only need Neo ID here
    // The initialization structs track producers/consumers for a given DFB and they would only be used by one of the unpacker or packer
    hartid = 8 + neo_id;
#else
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
#endif
    uint16_t hart_bit = 1 << hartid;

    uint32_t num_dfbs =
        local_dfb_mask;  // kernel config holds local_cb_mask but it gets hijacked to hold number of dfbs
    volatile uint8_t* base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);

    // each RISC populates its own g_dfb_interface entry
    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        // Read dfb_initializer_t (shared config)
        volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
        uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
        uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

        // Per-risc configs start after dfb_initializer_t
        volatile dfb_initializer_per_risc_t* per_risc_base =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

        // DPRINT << "hartid: 0x" << HEX() << hartid << " risc_mask: 0x" << risc_mask << " hart_bit: 0x" << hart_bit << DEC() << ENDL();
        if (risc_mask & hart_bit) {
            // Find this risc's per-risc config by counting set bits before this position
            uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & ((1 << hartid) - 1)));
            volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

            // Populate LocalDFBInterface from combined dfb_initializer_t + dfb_initializer_per_risc_t
            LocalDFBInterface& dfb_interface = g_dfb_interface[logical_dfb_id];

            // DPRINT << "risc_index: " << static_cast<uint32_t>(risc_index) << ENDL();
            dfb_interface.num_tcs_to_rr = per_risc_ptr->num_tcs_and_init.num_tcs_to_rr;
            // DPRINT << "num_tcs_to_rr: " << static_cast<uint32_t>(dfb_interface.num_tcs_to_rr) << ENDL();

            // Address fields are in bytes on host; convert to 16B units on TRISC (cb_addr_shift=4), keep bytes on DM
            // (cb_addr_shift=0)
            for (uint8_t i = 0; i < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; i++) {
                uint32_t base = per_risc_ptr->base_addr[i] >> cb_addr_shift;
                dfb_interface.tc_slots[i].base_addr = base;
                dfb_interface.tc_slots[i].limit = per_risc_ptr->limit[i] >> cb_addr_shift;
                dfb_interface.tc_slots[i].rd_ptr = base;
                dfb_interface.tc_slots[i].wr_ptr = base;
                dfb_interface.tc_slots[i].packed_tile_counter = per_risc_ptr->packed_tile_counter[i];
            }
            dfb_interface.entry_size = init_ptr->entry_size >> cb_addr_shift;
            // DPRINT << "entry_size: " << static_cast<uint32_t>(dfb_interface.entry_size) << ENDL();
            dfb_interface.stride_size_tiles = init_ptr->stride_in_entries;
            dfb_interface.stride_size = dfb_interface.entry_size * init_ptr->stride_in_entries;
            // DPRINT << "stride_size: " << static_cast<uint32_t>(dfb_interface.stride_size) << ENDL();
            dfb_interface.rd_entry_idx = 0;
            dfb_interface.wr_entry_idx = 0;
            dfb_interface.wr_entry_ptr = 0;

            dfb_interface.tc_idx = 0;
            dfb_interface.tensix_trisc_mask = init_ptr->risc_mask_bits.tensix_trisc_mask;
            dfb_interface.broadcast_tc = per_risc_ptr->num_tcs_and_init.broadcast_tc;

#ifndef COMPILE_FOR_TRISC
            if (per_risc_ptr->flags.is_producer) {
                dfb_interface.num_txn_ids = init_ptr->producer_txn_descriptor.num_txn_ids;
                dfb_interface.num_entries_per_txn_id = init_ptr->producer_txn_descriptor.num_entries_per_txn_id;
                dfb_interface.num_entries_per_txn_id_per_tc = init_ptr->producer_txn_descriptor.num_entries_per_txn_id_per_tc;
                for (uint8_t i = 0; i < dfb_interface.num_txn_ids; i++) {
                    dfb_interface.txn_ids[i] = init_ptr->producer_txn_descriptor.txn_ids[i];
                }
            } else {
                dfb_interface.num_txn_ids = init_ptr->consumer_txn_descriptor.num_txn_ids;
                dfb_interface.num_entries_per_txn_id = init_ptr->consumer_txn_descriptor.num_entries_per_txn_id;
                dfb_interface.num_entries_per_txn_id_per_tc = init_ptr->consumer_txn_descriptor.num_entries_per_txn_id_per_tc;
                for (uint8_t i = 0; i < dfb_interface.num_txn_ids; i++) {
                    dfb_interface.txn_ids[i] = init_ptr->consumer_txn_descriptor.txn_ids[i];
                }
            }
#endif
        }

        // Jump to next DFB: skip dfb_initializer_t + (num_riscs * dfb_initializer_per_risc_t)
        base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
    }

#ifndef COMPILE_FOR_TRISC
    // DM0 handles all remapper configuration and transaction id ISR initialization
    if (hartid == 0) {
        // Pass A: configure remapper for all DM producers across all DFBs, then enable
        bool enable_remapper = false;
        bool enable_implicit_sync = false;
        base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);
        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
            uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
            uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

            volatile dfb_initializer_per_risc_t* per_risc_base =
                reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

            uint8_t num_producer_tcs = 0;
            uint8_t producer_tcs[dfb::MAX_NUM_TILE_COUNTERS_TO_RR] = {};
            uint8_t num_consumer_tcs = 0;
            uint8_t consumer_tcs[dfb::MAX_NUM_TILE_COUNTERS_TO_RR] = {};
            for (uint8_t i = 0; i < num_riscs; i++) {
                volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + i;

                if (per_risc_ptr->flags.is_producer) {
                    if (per_risc_ptr->flags.remapper_en) { // Configure remapper for all producers with remapper_en (DM and Tensix)
                        enable_remapper = true;
                        uint8_t remapper_consumer_ids_mask = per_risc_ptr->remapper_consumer_ids_mask;
                        uint8_t producer_client_type = per_risc_ptr->producer_client_type;
                        uint8_t num_clientRs = static_cast<uint8_t>(__builtin_popcount(remapper_consumer_ids_mask));
                        uint8_t clientR_valid_mask = (1u << num_clientRs) - 1;
                        g_remapper_configurator.set_pair_index(static_cast<uint32_t>(per_risc_ptr->flags.remapper_pair_index));
                        // DPRINT << "Setting clientL fields clientL=" << static_cast<uint32_t>(producer_client_type)
                        //        << " tc: " << static_cast<uint32_t>(get_counter_id(per_risc_ptr->packed_tile_counter[0]))
                        //        << " mask: " << static_cast<uint32_t>(clientR_valid_mask) << ENDL();
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
                            // DPRINT << "Setting clientR slot " << static_cast<uint32_t>(clientR_idx)
                            //        << " id: " << static_cast<uint32_t>(id_R) << " tc: " << static_cast<uint32_t>(tc_R)
                            //        << ENDL();
                            g_remapper_configurator.set_clientR_slot(clientR_idx, id_R, tc_R);
                        }
                        // DPRINT << "Writing all remapper configs" << ENDL();
                        g_remapper_configurator.write_all_configs();
                    }

                    for (uint8_t i = 0; i < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; i++) {
                        producer_tcs[num_producer_tcs++] = per_risc_ptr->packed_tile_counter[i];
                    }
                } else {
                    for (uint8_t i = 0; i < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; i++) {
                        consumer_tcs[num_consumer_tcs++] = per_risc_ptr->packed_tile_counter[i];
                    }
                }
            }

            // Got all info from producers and consumers, now set up the TxnDFBDescriptor for producer and consumer
            uint32_t producer_txn_id_mask = 0;
            uint32_t consumer_txn_id_mask = 0;
            for (uint8_t i = 0; i < init_ptr->producer_txn_descriptor.num_txn_ids; i++) {
                uint8_t txn_id = init_ptr->producer_txn_descriptor.txn_ids[i];
                producer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = static_cast<uint8_t>(num_producer_tcs);
                for (uint8_t j = 0; j < dst.num_counters; j++) {
                    dst.tile_counters[j] = producer_tcs[j];
                }
                dst.tiles_to_post = init_ptr->producer_txn_descriptor.num_entries_per_txn_id_per_tc;
                SET_TILES_TO_PROCESS_THRES_TR_ACK(txn_id, init_ptr->producer_txn_descriptor.num_entries_to_process_threshold);
            }
            for (uint8_t i = 0; i < init_ptr->consumer_txn_descriptor.num_txn_ids; i++) {
                uint8_t txn_id = init_ptr->consumer_txn_descriptor.txn_ids[i];
                consumer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = static_cast<uint8_t>(num_consumer_tcs);
                for (uint8_t j = 0; j < dst.num_counters; j++) {
                    dst.tile_counters[j] = consumer_tcs[j];
                }
                dst.tiles_to_ack = init_ptr->consumer_txn_descriptor.num_entries_per_txn_id_per_tc;
                SET_TILES_TO_PROCESS_THRES_WR_SENT(txn_id, init_ptr->consumer_txn_descriptor.num_entries_to_process_threshold);
            }
            if (producer_txn_id_mask != 0) {
                enable_implicit_sync = true;
                // per_trid_tiles_to_process_set_interrupt_enable_cmdbuf_0(producer_txn_id_mask);
                uint64_t reg_val = CMDBUF_RD_REG(0, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);
                reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(producer_txn_id_mask & 0xFFFFFFFFULL) << 32);
                CMDBUF_WR_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);
            }
            if (consumer_txn_id_mask != 0) {
                enable_implicit_sync = true;
                // per_trid_wr_tiles_to_process_set_interrupt_enable_cmdbuf_0(consumer_txn_id_mask);
                uint64_t reg_val = CMDBUF_RD_REG(0, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);
                reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (consumer_txn_id_mask & 0xFFFFFFFFULL);
                CMDBUF_WR_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);
            }

            base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
        }

        if (enable_remapper) {
            // DPRINT << "Enabling remapper" << ENDL();
            g_remapper_configurator.enable_remapper();
        }

        if (enable_implicit_sync) {
            setup_isr_csrs();
        }
    }  // end if (hartid == 0)
#endif

    // Each DM and Tensix producer initialized their own TC
    base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);
    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
        uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
        uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

        volatile dfb_initializer_per_risc_t* per_risc_base =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

        if (risc_mask & hart_bit) {
            uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & ((1 << hartid) - 1)));
            volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

            if (per_risc_ptr->flags.is_producer) {
                while (per_risc_ptr->flags.remapper_en && !RemapperAPI::is_remapper_enabled());

                // Note: resetting tile counters does not reset the buffer capacity to 0
                for (uint8_t tc = 0; tc < per_risc_ptr->num_tcs_and_init.num_tcs_to_rr; tc++) {
                    dfb::PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                    uint8_t tc_id = dfb::get_counter_id(ptc);
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
                    // DPRINT << "dfb " << static_cast<uint32_t>(logical_dfb_id)
                    //         << " initializing tc_id: " << static_cast<uint32_t>(tc_id) << ENDL();
                    ckernel::trisc::tile_counters[tc_id].f.reset = 1;
                    ckernel::trisc::tile_counters[tc_id].f.buf_capacity = init_ptr->capacity;
#elif !defined(COMPILE_FOR_TRISC)
                    uint8_t tensix_id = dfb::get_tensix_id(ptc);
                    // DPRINT << "dfb " << static_cast<uint32_t>(logical_dfb_id)
                    //         << " initializing tc tensix_id: " << static_cast<uint32_t>(tensix_id)
                    //         << " tc_id: " << static_cast<uint32_t>(tc_id) << ENDL();
                    llk_intf_reset(tensix_id, tc_id);
                    llk_intf_set_capacity(tensix_id, tc_id, init_ptr->capacity);
#endif
                }
                // Single writer per per_risc entry; no atomic needed
                per_risc_ptr->num_tcs_and_init.tc_init_done = 1;
            }
        }

        base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
    }

    // After setting up g_dfb_interface, wait for all TCs to be initialized
    bool all_tcs_initialized = false;
    while (!all_tcs_initialized) {
        all_tcs_initialized = true;
        base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);

            uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
            uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

            volatile dfb_initializer_per_risc_t* per_risc_base =
                reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

            // Loop over per_risc: count producers that have set tc_init_done (each per_risc is separate cache line)
            uint8_t producers_done = 0;
            for (uint8_t i = 0; i < num_riscs; i++) {
                if (per_risc_base[i].flags.is_producer && per_risc_base[i].num_tcs_and_init.tc_init_done) {
                    producers_done++;
                }
            }
            all_tcs_initialized &= (producers_done == init_ptr->num_producers);

            base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
        }
    }
    // DPRINT << "all_tcs_initialized" << ENDL();
}
