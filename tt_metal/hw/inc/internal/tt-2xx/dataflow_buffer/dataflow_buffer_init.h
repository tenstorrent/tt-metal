// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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

// Participant mask (bits 0-11) only; storage u16 also includes tensix_trisc_mask in bits 12-15.
FORCE_INLINE uint16_t load_dfb_risc_mask(const volatile dfb_initializer_t* init) {
    const volatile uint8_t* bp =
        reinterpret_cast<const volatile uint8_t*>(init) + offsetof(dfb_initializer_t, risc_mask_bits);
    return static_cast<uint16_t>(bp[0]) | (static_cast<uint16_t>(bp[1] & 0x0Fu) << 8);
}

FORCE_INLINE uint8_t load_dfb_num_tcs_to_rr(const volatile dfb_initializer_per_risc_t* per_risc) {
    return reinterpret_cast<const volatile uint8_t*>(per_risc)[offsetof(dfb_initializer_per_risc_t, num_tcs_and_init)] &
           0x0Fu;
}

FORCE_INLINE uint8_t load_dfb_tc_init_done(const volatile dfb_initializer_per_risc_t* per_risc) {
    return (reinterpret_cast<const volatile uint8_t*>(per_risc)[offsetof(dfb_initializer_per_risc_t, num_tcs_and_init)] >>
            4) &
           1u;
}

FORCE_INLINE uint8_t load_dfb_per_risc_flags(const volatile dfb_initializer_per_risc_t* per_risc) {
    return reinterpret_cast<const volatile uint8_t*>(per_risc)[offsetof(dfb_initializer_per_risc_t, flags)];
}

// Poll until every DFB has all producer TCs initialized and DM0's ISR path marked done.
// Visits every DFB via host dfb_byte_offset[] (phase 1b — no layout stride walk).
FORCE_INLINE void wait_all_tcs_initialized(uint32_t tt_l1_ptr* dfb_config_base, uint32_t num_dfbs) {
    WAYPOINT("TCIW");
    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile uint16_t* dfb_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_byte_offset_table_byte_offset());

    bool all_tcs_initialized = false;
    while (!all_tcs_initialized) {
        all_tcs_initialized = true;

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            const uint32_t layout_byte_off = dfb_byte_offset_table[logical_dfb_id];
            volatile dfb_initializer_t* init_ptr =
                reinterpret_cast<volatile dfb_initializer_t*>(config_base + layout_byte_off);

            if (init_ptr->implicit_sync_configured != 1) {
                WAYPOINT("ISC");
                all_tcs_initialized = false;
                break;
            }

            const uint16_t risc_mask = load_dfb_risc_mask(init_ptr);
            const uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

            volatile dfb_initializer_per_risc_t* per_risc_base = reinterpret_cast<volatile dfb_initializer_per_risc_t*>(
                config_base + layout_byte_off + sizeof(dfb_initializer_t));

            int producers_done = 0;
            for (int i = 0; i < num_riscs; i++) {
                const uint8_t flags = load_dfb_per_risc_flags(&per_risc_base[i]);
                if ((flags & 0x80u) && load_dfb_tc_init_done(&per_risc_base[i])) {
                    WAYPOINT("PDI");
                    producers_done++;
                }
            }
            if (producers_done != init_ptr->num_producers) {
                WAYPOINT("PND");
                all_tcs_initialized = false;
                break;
            }
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
//   DM0 path:  dedicated contiguous blob loop (reads DM0 global blob directly — no dfb_initializer_t
//              fetches, no per-RISC cache pollution), followed by DM0's post-loop (remapper enable +
//              ISR setup + implicit_sync_configured write), followed by a TC-init loop for all DMs.
//   Other DMs: shared per-DFB layout loop (dfb_initializer_t + per-risc entries only).
//   TRISC path: shared per-DFB layout loop (same as other DMs).
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
    const uint8_t hart_u8 = static_cast<uint8_t>(hartid);

    uint32_t num_dfbs =
        local_dfb_mask;  // kernel config holds local_cb_mask but it gets hijacked to hold number of dfbs
    // DPRINT("num_dfbs: {}\n", num_dfbs);

    // Read the global header: region offsets, dfb_byte_offset[], per_risc_byte_offset[][], participation_mask[].
    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    volatile uint16_t* dfb_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_byte_offset_table_byte_offset());
    volatile uint16_t* per_risc_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_per_risc_byte_offset_table_byte_offset(static_cast<uint8_t>(num_dfbs)));
    uint32_t dm1_remapper_blob_offset = ghdr->dm1_remapper_blob_offset;
    uint32_t dm0_isr_blob_offset      = ghdr->dm0_isr_blob_offset;
    uint32_t per_dfb_layout_offset    = ghdr->per_dfb_layout_offset;

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    uint8_t compact_dfb_count = 0;
#endif

    // Timing probes shared by ALL RISCs (DMs + TRISCs). Zero = not reached.
    //   merged_sw     = t_after_merged_loop - start_time
    //   remapper_spin = accumulated producer spin before llk_intf (DM2-7 / TRISC prod)
    //   tc_hw         = accumulated llk_intf_reset/set_capacity or tile_counter HW
    //   wait_all      = t_after_wait_all - t_before_wait_all (barrier poll in wait_all_tcs_initialized)
    //   DM0 post-loop = isr_ie_writes, isr_enable, implicit_sync_stores (intervals printed at end)
    uint32_t t_after_merged_loop  = 0;
    uint32_t t_after_tc_init_loop = 0;
    uint32_t t_before_wait_all    = 0;
    uint32_t t_after_wait_all     = 0;
    // Producer-only: remapper spin vs TC HW (summed across all visited producer DFBs).
    uint32_t total_remapper_spin  = 0;
    uint32_t total_tc_hw          = 0;
    // First producer entry into remapper spin (absolute stamp; for debugging only).
    uint32_t t_before_tc_writes   = 0;

#ifndef COMPILE_FOR_TRISC
    // DM0-only: ISR txn mask and enable timing.
    uint32_t producer_txn_id_mask = 0;
    uint32_t consumer_txn_id_mask = 0;
    uint32_t end_isr_enable_time  = 0;
    uint32_t t_after_isr_ie_writes = 0;

    // DM1-only: remapper accumulator and timing.
    bool enable_remapper   = false;
    uint32_t remapper_hwm  = 0;   // one past the highest pair_index configured
    uint32_t end_remapper_config_time  = 0;
    uint32_t t_after_write_pairs_up_to = 0;

    // Granular timing probes. Per-DFB arrays: DM0 populates t_subpassA/B, DM1 populates t_rmp_pass.
    constexpr uint8_t MAX_PROBE_DFBS = 32;
    uint32_t t_subpassA[MAX_PROBE_DFBS]      = {};  // DM0: loop overhead per DFB (header + bulk-copy)
    uint32_t t_subpassB_desc[MAX_PROBE_DFBS] = {};  // DM0: after g_txn_dfb_descriptor populate (SW only)
    uint32_t t_subpassB[MAX_PROBE_DFBS]      = {};  // DM0: after CMDBUF threshold register writes (HW)
    uint32_t t_rmp_pass[MAX_PROBE_DFBS]      = {};  // DM1: after remapper slots processed per DFB (SW)
#endif

    // -----------------------------------------------------------------------
    // DM1: dedicated remapper blob loop. Runs in parallel with DM0's ISR blob loop.
    // Reads linearly from dm1_remapper_blob_offset — only remapper slot data,
    // no ISR/txn pollution, hardware prefetcher-friendly.
    // After all DFBs: burst-write remapper pairs, enable remapper.
    // -----------------------------------------------------------------------
#ifndef COMPILE_FOR_TRISC
    if (hartid == 1) {
        volatile tt_l1_ptr uint8_t* dm1_blob_ptr = config_base + dm1_remapper_blob_offset;

        // Local non-volatile staging buffer for one DFB's remapper entry.
        // Size = header(4) + max_rmp_slots(8×16) = 132 bytes = 33 words.
        WAYPOINT("RS");
        constexpr uint32_t MAX_DM1_ENTRY_WORDS =
            (sizeof(dfb_dm1_remapper_entry_header_t) +
             dfb::MAX_DM0_REMAPPER_SLOTS * sizeof(dfb_dm0_remapper_slot_t) + 3u) / 4u;
        uint32_t local_rmp[MAX_DM1_ENTRY_WORDS];

        // DPRINT("num_dfbs: {}\n", num_dfbs);
        // DPRINT("remapper starting");
        WAYPOINT("RS1");

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            // Bulk-copy this DFB's remapper entry into a non-volatile local buffer.
            const volatile tt_l1_ptr uint32_t* vsrc =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(dm1_blob_ptr);
            local_rmp[0] = vsrc[0];
            WAYPOINT("RS2");

            const dfb_dm1_remapper_entry_header_t* local_hdr =
                reinterpret_cast<const dfb_dm1_remapper_entry_header_t*>(local_rmp);
            int num_rmp = local_hdr->num_remapper_slots;
            WAYPOINT("RS3");

            uint32_t entry_bytes = sizeof(dfb_dm1_remapper_entry_header_t)
                                 + num_rmp * sizeof(dfb_dm0_remapper_slot_t);
            uint32_t entry_words = (entry_bytes + 3u) >> 2u;
            for (uint32_t w = 1u; w < entry_words; w++) {
                local_rmp[w] = vsrc[w];
            }
            WAYPOINT("RS4");

            const dfb_dm0_remapper_slot_t* slots =
                reinterpret_cast<const dfb_dm0_remapper_slot_t*>(
                    reinterpret_cast<const uint8_t*>(local_rmp) + sizeof(dfb_dm1_remapper_entry_header_t));

            for (int s = 0; s < num_rmp; s++) {
                const dfb_dm0_remapper_slot_t& slot = slots[s];
                enable_remapper = true;
                g_remapper_configurator.load_pair_raw(
                    static_cast<uint32_t>(slot.pair_index), slot.clientR_val, slot.clientL_val);
                uint32_t hwm = slot.pair_index + 1u;
                if (hwm > remapper_hwm) { remapper_hwm = hwm; }
            }
            t_rmp_pass[logical_dfb_id] = rdcycle();

            dm1_blob_ptr += entry_bytes;
        }

        // Both DM and Neo producers spin on is_remapper_enabled() before initializing tile counters
        if (enable_remapper) {
            g_remapper_configurator.write_pairs_up_to(remapper_hwm);
            t_after_write_pairs_up_to = rdcycle();
            g_remapper_configurator.enable_remapper();
            end_remapper_config_time = rdcycle();
        }

        WAYPOINT("RSD");
    }
    // -----------------------------------------------------------------------
    // DM0: dedicated ISR blob loop. Runs in parallel with DM1's remapper blob loop.
    // Reads linearly from dm0_isr_blob_offset — only CMDBUF/ISR txn data,
    // no remapper/init pollution, hardware prefetcher-friendly.
    // -----------------------------------------------------------------------
    else if (hartid == 0) {
        volatile tt_l1_ptr uint8_t* dm0_blob_ptr = config_base + dm0_isr_blob_offset;

        // Local non-volatile staging buffer for one DFB's ISR entry.
        // Size = header(4) + max_txn_entries(2×4×16) = 132 bytes = 33 words.
        constexpr uint32_t MAX_DM0_ISR_ENTRY_WORDS =
            (sizeof(dfb_dm0_isr_entry_header_t) +
             2u * dfb::NUM_TXN_IDS * sizeof(dfb_dm0_txn_entry_t) + 3u) / 4u;
        uint32_t local_isr[MAX_DM0_ISR_ENTRY_WORDS];

        // DPRINT("ISR starting");

        WAYPOINT("IS1");

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            t_subpassA[logical_dfb_id] = rdcycle();

            const volatile tt_l1_ptr uint32_t* vsrc =
                reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(dm0_blob_ptr);
            local_isr[0] = vsrc[0];
            WAYPOINT("IS2");

            const dfb_dm0_isr_entry_header_t* local_hdr =
                reinterpret_cast<const dfb_dm0_isr_entry_header_t*>(local_isr);
            uint8_t num_prod = local_hdr->num_producer_txns;
            uint8_t num_cons = local_hdr->num_consumer_txns;

            WAYPOINT("IS3");

            uint32_t entry_bytes = sizeof(dfb_dm0_isr_entry_header_t)
                                 + (num_prod + num_cons) * sizeof(dfb_dm0_txn_entry_t);
            uint32_t entry_words = (entry_bytes + 3u) >> 2u;
            for (uint32_t w = 1u; w < entry_words; w++) {
                local_isr[w] = vsrc[w];
            }

            WAYPOINT("IS4");

            const dfb_dm0_txn_entry_t* prod_txns =
                reinterpret_cast<const dfb_dm0_txn_entry_t*>(
                    reinterpret_cast<const uint8_t*>(local_isr) + sizeof(dfb_dm0_isr_entry_header_t));
            const dfb_dm0_txn_entry_t* cons_txns = prod_txns + num_prod;

            WAYPOINT("IS5");

            for (int i = 0; i < num_prod; i++) {
                const dfb_dm0_txn_entry_t& e = prod_txns[i];
                const uint32_t txn_id        = e.txn_id;
                const int      num_tcs       = e.num_tcs;
                const uint32_t tiles_to_post = e.tiles_to_post_or_ack;
                producer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = num_tcs;
                for (int j = 0; j < num_tcs; j++) {
                    dst.tile_counters[j] = e.tile_counters[j];
                }
                dst.tiles_to_post = tiles_to_post;
            }
            for (int i = 0; i < num_cons; i++) {
                const dfb_dm0_txn_entry_t& e = cons_txns[i];
                const uint32_t txn_id       = e.txn_id;
                const int      num_tcs      = e.num_tcs;
                const uint32_t tiles_to_ack = e.tiles_to_post_or_ack;
                consumer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = num_tcs;
                for (int j = 0; j < num_tcs; j++) {
                    dst.tile_counters[j] = e.tile_counters[j];
                }
                dst.tiles_to_ack = tiles_to_ack;
            }
            t_subpassB_desc[logical_dfb_id] = rdcycle();

            for (int i = 0; i < num_prod; i++) {
                const dfb_dm0_txn_entry_t& e = prod_txns[i];
                const uint32_t txn_id    = e.txn_id;
                const uint32_t threshold = e.threshold;
                CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, txn_id);
                asm volatile("nop");
                SET_TILES_TO_PROCESS_THRES_TR_ACK(txn_id, threshold);
            }
            for (int i = 0; i < num_cons; i++) {
                const dfb_dm0_txn_entry_t& e = cons_txns[i];
                const uint32_t txn_id    = e.txn_id;
                const uint32_t threshold = e.threshold;
                CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, txn_id);
                asm volatile("nop");
                SET_TILES_TO_PROCESS_THRES_WR_SENT(txn_id, threshold);
            }
            t_subpassB[logical_dfb_id] = rdcycle();

            dm0_blob_ptr += entry_bytes;
        }

        uint64_t reg_val = CMDBUF_RD_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(producer_txn_id_mask & 0xFFFFFFFFULL) << 32);
        CMDBUF_WR_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);

        reg_val = CMDBUF_RD_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (consumer_txn_id_mask & 0xFFFFFFFFULL);
        CMDBUF_WR_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);
        t_after_isr_ie_writes = rdcycle();

        if ((producer_txn_id_mask | consumer_txn_id_mask) != 0) {
            enable_dfb_tile_isr();
        } else {
            disable_dfb_tile_isr();
        }
        end_isr_enable_time = rdcycle();

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            const uint32_t layout_byte_off = dfb_byte_offset_table[logical_dfb_id];
            volatile dfb_initializer_t* init_ptr =
                reinterpret_cast<volatile dfb_initializer_t*>(config_base + layout_byte_off);
            init_ptr->implicit_sync_configured = 1;
        }



        WAYPOINT("ISD");
    } else
#endif  // !COMPILE_FOR_TRISC
    // -----------------------------------------------------------------------
    // DM2-7 + TRISC: shared per-DFB layout loop.
    // Populates g_dfb_interface (TC base/limit addrs, txn IDs, entry sizes) for
    // each RISC that participates as a producer or consumer.
    // DM0 and DM1 are pure coordinators and skip this loop entirely.
    // Host guarantees participation_mask[hartid] is well-formed (no bits >= num_dfbs).
    // -----------------------------------------------------------------------
    {
        uint32_t participating = ghdr->participation_mask[hart_u8];
        while (participating) {
            const uint32_t logical_dfb_id = __builtin_ctz(participating);
            // DPRINT("participating: {}\n", participating);
            participating &= participating - 1u;
            // DPRINT("logical_dfb_id: {}\n", logical_dfb_id);
            WAYPOINT("L1");

            const uint32_t layout_byte_off = dfb_byte_offset_table[logical_dfb_id];
            volatile dfb_initializer_t* init_ptr =
                reinterpret_cast<volatile dfb_initializer_t*>(config_base + layout_byte_off);

            WAYPOINT("L2");

            const uint32_t per_risc_table_idx =
                dfb_per_risc_byte_offset_table_index(static_cast<uint8_t>(logical_dfb_id), hart_u8);
            const uint32_t per_risc_byte_off = per_risc_byte_offset_table[per_risc_table_idx];
            volatile dfb_initializer_per_risc_t* per_risc_ptr =
                reinterpret_cast<volatile dfb_initializer_per_risc_t*>(config_base + per_risc_byte_off);

            WAYPOINT("L3");

            // --- Sub-pass A: populate g_dfb_interface ---
            {

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
                ASSERT(compact_dfb_count < dfb::MAX_ACTIVE_DFBS_PACK);
                const uint8_t compact_dfb_id = compact_dfb_count++;
                g_dfb_logical_to_compact[logical_dfb_id] = compact_dfb_id;
                LocalDFBInterface& dfb_interface = g_dfb_interface[compact_dfb_id];
#else
                LocalDFBInterface& dfb_interface = g_dfb_interface[logical_dfb_id];
#endif

                WAYPOINT("L4");
                const uint8_t num_tcs = load_dfb_num_tcs_to_rr(per_risc_ptr);
                dfb_interface.num_tcs_to_rr = num_tcs;
                // DPRINT("dfb_interface.num_tcs_to_rr: {}\n", dfb_interface.num_tcs_to_rr);
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

                // DPRINT("got here");
                WAYPOINT("L5");

                for (int i = 0; i < num_tcs; i++) {
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
                    // DPRINT("dfb_interface.tc_slots[i].base_addr: {}\n", dfb_interface.tc_slots[i].base_addr);
#endif
                }

                WAYPOINT("L6");

                dfb_interface.tc_idx = 0;
#ifndef COMPILE_FOR_TRISC
                dfb_interface.broadcast_tc =
                    (reinterpret_cast<const volatile uint8_t*>(per_risc_ptr)[offsetof(
                         dfb_initializer_per_risc_t, num_tcs_and_init)] >>
                     5) &
                    1u;

                if (load_dfb_per_risc_flags(per_risc_ptr) & 0x80u) {
                    dfb_interface.num_txn_ids = init_ptr->producer_txn_descriptor.num_txn_ids;
                    dfb_interface.threshold = init_ptr->producer_txn_descriptor.num_entries_to_process_threshold;
                    dfb_interface.num_entries_per_txn_id = init_ptr->producer_txn_descriptor.num_entries_per_txn_id;
                    dfb_interface.num_entries_per_txn_id_per_tc =
                        init_ptr->producer_txn_descriptor.num_entries_per_txn_id_per_tc;
                    for (int i = 0; i < dfb_interface.num_txn_ids; i++) {
                        dfb_interface.txn_ids[i] = init_ptr->producer_txn_descriptor.txn_ids[i];
                    }
                } else {
                    dfb_interface.num_txn_ids = init_ptr->consumer_txn_descriptor.num_txn_ids;
                    dfb_interface.threshold = init_ptr->consumer_txn_descriptor.num_entries_to_process_threshold;
                    dfb_interface.num_entries_per_txn_id = init_ptr->consumer_txn_descriptor.num_entries_per_txn_id;
                    dfb_interface.num_entries_per_txn_id_per_tc =
                        init_ptr->consumer_txn_descriptor.num_entries_per_txn_id_per_tc;
                    for (int i = 0; i < dfb_interface.num_txn_ids; i++) {
                        dfb_interface.txn_ids[i] = init_ptr->consumer_txn_descriptor.txn_ids[i];
                    }
                }
#endif

                WAYPOINT("L7");

                // --- TC hardware init merged into this loop (producers only) ---
                if (load_dfb_per_risc_flags(per_risc_ptr) & 0x80u) {
                    if (load_dfb_per_risc_flags(per_risc_ptr) & 0x40u) {
                        if (!t_before_tc_writes) {
                            t_before_tc_writes = rdcycle();
                        }
                        const uint32_t spin_start = rdcycle();
                        while ((load_dfb_per_risc_flags(per_risc_ptr) & 0x40u) &&
                               !overlay::RemapperAPI::is_remapper_enabled());
                        total_remapper_spin += rdcycle() - spin_start;
                    }
                    const uint32_t tc_hw_start = rdcycle();
                    for (int tc = 0; tc < load_dfb_num_tcs_to_rr(per_risc_ptr); tc++) {
                        dfb::PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                        uint8_t tc_id = dfb::get_counter_id(ptc);
#ifndef COMPILE_FOR_TRISC
                        uint8_t tensix_id = dfb::get_tensix_id(ptc);
                        overlay::llk_intf_reset(tensix_id, tc_id);
                        overlay::llk_intf_set_capacity(tensix_id, tc_id, init_ptr->capacity);
#elif defined(UCK_CHLKC_PACK)
                        ckernel::trisc::tile_counters[tc_id].f.reset = 1;
                        ckernel::trisc::tile_counters[tc_id].f.buf_capacity = init_ptr->capacity;
#endif
                    }
                    total_tc_hw += rdcycle() - tc_hw_start;
                    per_risc_ptr->num_tcs_and_init.tc_init_done = 1;
                }
            }
            WAYPOINT("L8");
        }
    }

    // -----------------------------------------------------------------------
    // End merged loops (DM0 blob loop + shared per-DFB layout loop)
    // -----------------------------------------------------------------------

    // Point 4: all RISCs (DMs + TRISCs) capture merged loop end.
    // SW cost = t_after_merged_loop - start_time.
    t_after_merged_loop = rdcycle();

    // Point 6: interval below is DM0 implicit_sync stores only (producer TC HW is in merged loop).
    t_after_tc_init_loop = rdcycle();


    t_before_wait_all = rdcycle();
    wait_all_tcs_initialized(dfb_config_base, num_dfbs);
    t_after_wait_all = rdcycle();
    WAYPOINT("L12");
    uint32_t end_time = rdcycle();

    // All DPRINTs deferred to here so zero DPRINT overhead falls inside the timed region.
    const uint32_t merged_sw = t_after_merged_loop ? (t_after_merged_loop - start_time) : 0;
    const uint32_t wait_all_cycles = t_after_wait_all - t_before_wait_all;

    DPRINT("start_time: {}\n", start_time);
    DPRINT("timing: merged_sw={}\n", merged_sw);
    if (total_remapper_spin) {
        DPRINT("timing: remapper_spin={}\n", total_remapper_spin);
    }
    if (total_tc_hw) {
        DPRINT("timing: tc_hw={}\n", total_tc_hw);
    }
    DPRINT("timing: wait_all={}\n", wait_all_cycles);
#ifndef COMPILE_FOR_TRISC
    if (hartid == 0) {
        uint32_t total_subpassA     = 0;
        uint32_t total_subpassB_desc = 0;
        uint32_t total_subpassB_hw  = 0;
        for (uint32_t i = 0; i < num_dfbs; i++) {
            uint32_t prev_end      = (i == 0) ? start_time : t_subpassB[i - 1];
            uint32_t elapsed_a     = t_subpassA[i] - prev_end;
            uint32_t elapsed_desc  = t_subpassB_desc[i] - t_subpassA[i];
            uint32_t elapsed_hw    = t_subpassB[i] - t_subpassB_desc[i];
            total_subpassA      += elapsed_a;
            total_subpassB_desc += elapsed_desc;
            total_subpassB_hw   += elapsed_hw;
            DPRINT("DFB{} subpassA={} subpassB_desc={} subpassB_hw={}\n", i, elapsed_a, elapsed_desc, elapsed_hw);
        }
        DPRINT("total_subpassA={} total_subpassB_desc={} total_subpassB_hw={}\n",
               total_subpassA, total_subpassB_desc, total_subpassB_hw);

        if (t_after_merged_loop && t_after_isr_ie_writes > t_after_merged_loop) {
            DPRINT("timing: isr_ie_writes={}\n", t_after_isr_ie_writes - t_after_merged_loop);
        }
        if (end_isr_enable_time && t_after_isr_ie_writes) {
            DPRINT("timing: isr_enable={}\n", end_isr_enable_time - t_after_isr_ie_writes);
        }
        if (t_after_tc_init_loop && end_isr_enable_time) {
            DPRINT("timing: implicit_sync_stores={}\n", t_after_tc_init_loop - end_isr_enable_time);
        } else if (t_after_tc_init_loop && t_after_isr_ie_writes) {
            DPRINT("timing: implicit_sync_stores={}\n", t_after_tc_init_loop - t_after_isr_ie_writes);
        }
    }
    if (hartid == 1) {
        uint32_t total_rmp = 0;
        for (uint32_t i = 0; i < num_dfbs; i++) {
            uint32_t prev_end = (i == 0) ? start_time : t_rmp_pass[i - 1];
            uint32_t elapsed  = t_rmp_pass[i] - prev_end;
            total_rmp += elapsed;
            DPRINT("DFB{} rmp_pass={}\n", i, elapsed);
        }
        DPRINT("total_rmp={}\n", total_rmp);

        if (t_after_write_pairs_up_to && t_after_merged_loop) {
            DPRINT("timing: write_pairs_hw={}\n", t_after_write_pairs_up_to - t_after_merged_loop);
        }
        if (end_remapper_config_time && t_after_write_pairs_up_to) {
            DPRINT("timing: enable_remapper_hw={}\n", end_remapper_config_time - t_after_write_pairs_up_to);
        }
    }
    if (t_after_tc_init_loop) {
        DPRINT("t_after_tc_init_loop: {}\n", t_after_tc_init_loop);
    }
#endif
    if (t_after_merged_loop) {
        DPRINT("t_after_merged_loop: {}\n", t_after_merged_loop);
    }
    DPRINT("t_before_wait_all: {}\n", t_before_wait_all);
    DPRINT("t_after_wait_all: {}\n", t_after_wait_all);
    DPRINT("end_time: {}\n", end_time);
    DPRINT("timing: e2e={}\n", end_time - start_time);
}
