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
#endif

#include "api/debug/dprint.h"

// Global DFB interface array - defined in firmware, declared here for use by setup functions
// For kernels (NCRISC/BRISC/TRISC), provide a definition since they're compiled separately
extern thread_local ::experimental::LocalDFBInterface g_dfb_interface[32];
#ifndef COMPILE_FOR_TRISC
extern RemapperAPI g_remapper_configurator;
#endif

namespace experimental {

FORCE_INLINE void setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t local_dfb_mask) {
    uint64_t hartid;
#ifdef COMPILE_FOR_TRISC
    std::uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    std::uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
    hartid = 8 + 4 * neo_id + trisc_id;  // after 8 DM cores
#else
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
#endif
    uint16_t hart_bit = 1 << hartid;

    uint32_t num_dfbs =
        local_dfb_mask;  // kernel config holds local_cb_mask but it gets hijacked to hold number of dfbs
    volatile uint8_t* base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);

#ifndef COMPILE_FOR_TRISC
    bool enable_remapper = false;  // if remapper used once then needs to be globally set
#endif

    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        // Read dfb_initializer_t (shared config)
        volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);

        uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
        uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

        // Per-risc configs start after dfb_initializer_t
        volatile dfb_initializer_per_risc_t* per_risc_base =
            reinterpret_cast<volatile dfb_initializer_per_risc_t*>(base_ptr + sizeof(dfb_initializer_t));

        DPRINT << "hartid: 0x" << HEX() << hartid << " risc_mask: 0x" << HEX() << risc_mask << " hart_bit: 0x" << HEX()
               << hart_bit << DEC() << ENDL();
        if (risc_mask & hart_bit) {
            // Find this risc's per-risc config by counting set bits before this position
            uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & ((1 << hartid) - 1)));
            volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

            // Populate LocalDFBInterface from combined dfb_initializer_t + dfb_initializer_per_risc_t
            LocalDFBInterface& dfb_interface = ::g_dfb_interface[logical_dfb_id];

            DPRINT << "risc_index: " << static_cast<uint32_t>(risc_index) << ENDL();
            dfb_interface.num_tcs_to_rr = per_risc_ptr->num_tcs_to_rr;
            DPRINT << "num_tcs_to_rr: " << static_cast<uint32_t>(dfb_interface.num_tcs_to_rr) << ENDL();

            // Copy per-risc fields
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
                       << "]: " << (uint32_t)get_tensix_id((uint32_t)dfb_interface.packed_tile_counter[i]) << " "
                       << (uint32_t)get_counter_id((uint32_t)dfb_interface.packed_tile_counter[i]) << ENDL();
            }
            dfb_interface.entry_size = init_ptr->entry_size;
            DPRINT << "entry_size: " << static_cast<uint32_t>(dfb_interface.entry_size) << ENDL();
            dfb_interface.stride_size = init_ptr->stride_size;
            DPRINT << "stride_size: " << static_cast<uint32_t>(dfb_interface.stride_size) << ENDL();

            dfb_interface.num_txn_ids = init_ptr->num_txn_ids;
            for (uint8_t i = 0; i < init_ptr->num_txn_ids; i++) {
                dfb_interface.txn_ids[i] = init_ptr->txn_ids[i];
            }
            dfb_interface.num_entries_per_txn_id = init_ptr->num_entries_per_txn_id;
            dfb_interface.num_entries_per_txn_id_per_tc = init_ptr->num_entries_per_txn_id_per_tc;

            dfb_interface.remapper_pair_index = per_risc_ptr->flags.remapper_pair_index;
            DPRINT << "remapper_pair_index: " << static_cast<uint32_t>(dfb_interface.remapper_pair_index) << ENDL();

#ifndef COMPILE_FOR_TRISC
            // Configure remapper if needed (must be done before TC init)
            if (per_risc_ptr->flags.should_init_tc && per_risc_ptr->flags.remapper_en) {
                if (risc_index == 0) {  // update this
                    enable_remapper = true;
                }
                // remapper_consumer_ids_mask is a bitmask of clientTypes (id_R) for BLOCKED consumers
                uint8_t remapper_consumer_ids_mask = init_ptr->remapper_consumer_ids_mask;
                uint8_t num_clientRs = static_cast<uint8_t>(__builtin_popcount(remapper_consumer_ids_mask));
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
                // Program each consumer's R slot by extracting set bits from mask
                uint8_t mask_remaining = remapper_consumer_ids_mask;
                for (uint8_t clientR_idx = 0; clientR_idx < num_clientRs; clientR_idx++) {
                    // Extract id_R: position of lowest set bit in remaining mask = clientType
                    uint8_t id_R = static_cast<uint8_t>(__builtin_ctz(mask_remaining));
                    mask_remaining &= mask_remaining - 1;  // Clear lowest set bit
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

#ifndef COMPILE_FOR_TRISC
    // all DFBs were initialized, safe to enable remapper if used
    if (enable_remapper && hartid == 0) {  // update how one risc enables the remapper
        DPRINT << "Enabling remapper" << ENDL();
        g_remapper_configurator.enable_remapper();
    }

    // Initialize TCs after remapper is enabled - only the RISC marked as responsible should do this
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
    DPRINT << "Wait for all DFB TCs to be initialized init_ptr is " << HEX() << (uintptr_t)(dfb_config_base) << DEC()
           << ENDL();
    bool all_tcs_initialized = false;
    while (!all_tcs_initialized) {
        all_tcs_initialized = true;
        base_ptr = reinterpret_cast<volatile uint8_t*>(dfb_config_base);

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
            uint16_t risc_mask = (init_ptr->risc_mask_bits.tensix_mask << 8) | init_ptr->risc_mask_bits.dm_mask;
            uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));
            // DPRINT << "risc_mask: " << HEX() << risc_mask << ENDL();

            // TODO: Ring buffer is in uncached region so its okay to poll value. Needs to be uplifted when caching is
            // supported
            asm("FENCE.i");
            all_tcs_initialized &= init_ptr->risc_mask_bits.tc_initialized;
            // DPRINT << "all_tcs_initialized: " << uint32_t(init_ptr->risc_mask_bits.tc_initialized) << ENDL();

            base_ptr += sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t));
        }
    }
    DPRINT << "DFBs initialized" << ENDL();
}

}  // namespace experimental
