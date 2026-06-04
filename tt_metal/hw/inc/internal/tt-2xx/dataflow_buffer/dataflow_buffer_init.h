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

FORCE_INLINE uint32_t dfb_l1_load_u32(volatile tt_l1_ptr uint8_t* base, uint32_t byte_off) {
    const volatile tt_l1_ptr uint8_t* p = base + byte_off;
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
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
// shared_layout_ptr points to the first dfb_initializer_t (= dfb_config_base + per_dfb_layout_offset).
// The DM0 global blob is no longer interleaved, so the per-DFB stride is simply
//   sizeof(dfb_initializer_t) + num_riscs * sizeof(dfb_initializer_per_risc_t).
FORCE_INLINE void wait_all_tcs_initialized(
    uint32_t tt_l1_ptr* dfb_config_base, uint32_t layout_start_off, uint32_t num_dfbs, uint64_t hartid) {
    WAYPOINT("TCIW");
    bool all_tcs_initialized = false;
    while (!all_tcs_initialized) {
        all_tcs_initialized = true;
        uint32_t layout_cursor = layout_start_off;

        volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            volatile dfb_initializer_t* init_ptr =
                reinterpret_cast<volatile dfb_initializer_t*>(config_base + layout_cursor);

            if (init_ptr->implicit_sync_configured != 1) {
                all_tcs_initialized = false;
                break;
            }

            const uint16_t risc_mask = load_dfb_risc_mask(init_ptr);
            uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

            volatile dfb_initializer_per_risc_t* per_risc_base = reinterpret_cast<volatile dfb_initializer_per_risc_t*>(
                config_base + layout_cursor + sizeof(dfb_initializer_t));

            int producers_done = 0;
            for (int i = 0; i < num_riscs; i++) {
                const uint8_t flags = load_dfb_per_risc_flags(&per_risc_base[i]);
                if ((flags & 0x80u) && load_dfb_tc_init_done(&per_risc_base[i])) {
                    producers_done++;
                }
            }
            if (producers_done != init_ptr->num_producers) {
                all_tcs_initialized = false;
                break;
            }

            layout_cursor += static_cast<uint32_t>(
                sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t)));
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
    const uint16_t hart_bit = (hart_u8 < 16) ? static_cast<uint16_t>(1u << hart_u8) : 0;
    const uint32_t risc_prefix_mask =
        hart_u8 >= 32 ? 0u : static_cast<uint32_t>((1ULL << hart_u8) - 1ULL);

    uint32_t num_dfbs =
        local_dfb_mask;  // kernel config holds local_cb_mask but it gets hijacked to hold number of dfbs

    // Read the global header: region offsets, dfb_byte_offset[], participation_mask[].
    // Layout: [dfb_global_header_t(64B) + uint16_t dfb_byte_offset[num_dfbs]] [DM1 blob] [DM0 blob] [layouts]
    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    volatile uint16_t* dfb_byte_offset_table = reinterpret_cast<volatile uint16_t*>(
        config_base + dfb_byte_offset_table_byte_offset());
    uint32_t dm1_remapper_blob_offset = ghdr->dm1_remapper_blob_offset;
    uint32_t dm0_isr_blob_offset      = ghdr->dm0_isr_blob_offset;
    uint32_t per_dfb_layout_offset    = ghdr->per_dfb_layout_offset;

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    uint8_t compact_dfb_count = 0;
#endif

    // Timing probes shared by ALL RISCs (DMs + TRISCs). Zero = not reached.
    //   SW cost  = t_after_merged_loop - start_time   (for every RISC)
    //   wait cost= end_time - t_after_tc_init_loop    (DMs)
    //            = end_time - t_after_merged_loop      (TRISCs, which skip the TC init loop)
    uint32_t t_after_merged_loop  = 0;
    uint32_t t_after_tc_init_loop = 0;

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
    // t_before_tc_writes set by all producer DMs to split spin-wait from TC HW writes.
    constexpr uint8_t MAX_PROBE_DFBS = 32;
    uint32_t t_subpassA[MAX_PROBE_DFBS]   = {};  // DM0: loop overhead per DFB (no dfb_initializer_t fetch)
    uint32_t t_subpassB[MAX_PROBE_DFBS]   = {};  // DM0: after CMDBUF threshold writes per DFB
    uint32_t t_rmp_pass[MAX_PROBE_DFBS]   = {};  // DM1: after remapper slots processed per DFB
    uint32_t t_before_tc_writes = 0;             // all producer DMs: before first llk_intf_reset
#endif

    // -----------------------------------------------------------------------
    // DM1: dedicated remapper blob loop. Runs in parallel with DM0's ISR blob loop.
    // Reads linearly from dm1_remapper_blob_offset — only remapper slot data,
    // no ISR/txn pollution, hardware prefetcher-friendly.
    // After all DFBs: burst-write remapper pairs, enable remapper.
    // -----------------------------------------------------------------------
#ifndef COMPILE_FOR_TRISC
    if (hartid == 1) {
        uint32_t dm1_blob_cursor = dm1_remapper_blob_offset;

        // Local non-volatile staging buffer for one DFB's remapper entry.
        // Size = header(4) + max_rmp_slots(8×16) = 132 bytes = 33 words.
        constexpr uint32_t MAX_DM1_ENTRY_WORDS =
            (sizeof(dfb_dm1_remapper_entry_header_t) +
             dfb::MAX_DM0_REMAPPER_SLOTS * sizeof(dfb_dm0_remapper_slot_t) + 3u) / 4u;
        uint32_t local_rmp[MAX_DM1_ENTRY_WORDS];

        // DPRINT("num_dfbs: {}\n", num_dfbs);

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            // Bulk-copy this DFB's remapper entry into a non-volatile local buffer.
            // DPRINT("here");
            local_rmp[0] = dfb_l1_load_u32(config_base, dm1_blob_cursor);
            // DPRINT("local_rmp[0]: {}\n", local_rmp[0]);

            const dfb_dm1_remapper_entry_header_t* local_hdr =
                reinterpret_cast<const dfb_dm1_remapper_entry_header_t*>(local_rmp);
            int num_rmp = local_hdr->num_remapper_slots;
            // DPRINT("num_rmp: {}\n", num_rmp);

            uint32_t entry_bytes = sizeof(dfb_dm1_remapper_entry_header_t)
                                 + num_rmp * sizeof(dfb_dm0_remapper_slot_t);
            uint32_t entry_words = (entry_bytes + 3u) >> 2u;
            for (uint32_t w = 1u; w < entry_words; w++) {
                local_rmp[w] = dfb_l1_load_u32(config_base, dm1_blob_cursor + (w << 2));
            }

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

            dm1_blob_cursor += entry_bytes;
        }
    }
    // -----------------------------------------------------------------------
    // DM0: dedicated ISR blob loop. Runs in parallel with DM1's remapper blob loop.
    // Reads linearly from dm0_isr_blob_offset — only CMDBUF/ISR txn data,
    // no remapper/init pollution, hardware prefetcher-friendly.
    // -----------------------------------------------------------------------
    else if (hartid == 0) {
        uint32_t dm0_blob_cursor = dm0_isr_blob_offset;

        // Local non-volatile staging buffer for one DFB's ISR entry.
        // Size = header(4) + max_txn_entries(2×4×16) = 132 bytes = 33 words.
        constexpr uint32_t MAX_DM0_ISR_ENTRY_WORDS =
            (sizeof(dfb_dm0_isr_entry_header_t) +
             2u * dfb::NUM_TXN_IDS * sizeof(dfb_dm0_txn_entry_t) + 3u) / 4u;
        uint32_t local_isr[MAX_DM0_ISR_ENTRY_WORDS];

        // DPRINT("dm0_blob_cursor: {}\n", dm0_blob_cursor);

        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            t_subpassA[logical_dfb_id] = rdcycle();

            // DPRINT("here!");
            local_isr[0] = dfb_l1_load_u32(config_base, dm0_blob_cursor);
            // DPRINT("local_isr[0]: {}\n", local_isr[0]);

            const dfb_dm0_isr_entry_header_t* local_hdr =
                reinterpret_cast<const dfb_dm0_isr_entry_header_t*>(local_isr);
            uint8_t num_prod = local_hdr->num_producer_txns;
            uint8_t num_cons = local_hdr->num_consumer_txns;
            // DPRINT("num_prod: {}, num_cons: {}\n", num_prod, num_cons);

            uint32_t entry_bytes = sizeof(dfb_dm0_isr_entry_header_t)
                                 + (num_prod + num_cons) * sizeof(dfb_dm0_txn_entry_t);
            uint32_t entry_words = (entry_bytes + 3u) >> 2u;
            for (uint32_t w = 1u; w < entry_words; w++) {
                local_isr[w] = dfb_l1_load_u32(config_base, dm0_blob_cursor + (w << 2));
            }

            const dfb_dm0_txn_entry_t* prod_txns =
                reinterpret_cast<const dfb_dm0_txn_entry_t*>(
                    reinterpret_cast<const uint8_t*>(local_isr) + sizeof(dfb_dm0_isr_entry_header_t));
            const dfb_dm0_txn_entry_t* cons_txns = prod_txns + num_prod;

            for (int i = 0; i < num_prod; i++) {
                const dfb_dm0_txn_entry_t& e = prod_txns[i];
                // Hoist all fields used after (or inside) the TC copy inner loop into
                // dedicated uint32_t locals. This prevents the compiler from clobbering
                // the source register (e.g. a2 used for txn_id then overwritten by
                // txn_id<<5 for the descriptor index) and reloading from memory for each
                // ROCC instruction and the loop guard — eliminating 3 redundant lbu loads
                // per transaction.
                const uint32_t txn_id        = e.txn_id;
                const int      num_tcs       = e.num_tcs;
                const uint32_t threshold     = e.threshold;
                const uint32_t tiles_to_post = e.tiles_to_post_or_ack;
                producer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = num_tcs;
                for (int j = 0; j < num_tcs; j++) {
                    dst.tile_counters[j] = e.tile_counters[j];
                }
                dst.tiles_to_post = tiles_to_post;
                CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, txn_id);
                asm volatile("nop");
                SET_TILES_TO_PROCESS_THRES_TR_ACK(txn_id, threshold);
            }
            for (int i = 0; i < num_cons; i++) {
                const dfb_dm0_txn_entry_t& e = cons_txns[i];
                const uint32_t txn_id       = e.txn_id;
                const int      num_tcs      = e.num_tcs;
                const uint32_t threshold    = e.threshold;
                const uint32_t tiles_to_ack = e.tiles_to_post_or_ack;
                consumer_txn_id_mask |= (1u << txn_id);
                volatile TxnDFBDescriptor& dst = g_txn_dfb_descriptor[txn_id];
                dst.num_counters = num_tcs;
                for (int j = 0; j < num_tcs; j++) {
                    dst.tile_counters[j] = e.tile_counters[j];
                }
                CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, txn_id);
                asm volatile("nop");
                dst.tiles_to_ack = tiles_to_ack;
                SET_TILES_TO_PROCESS_THRES_WR_SENT(txn_id, threshold);
            }
            t_subpassB[logical_dfb_id] = rdcycle();

            dm0_blob_cursor += entry_bytes;
        }
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
            participating &= participating - 1u;
            // DPRINT("participating: {}\n", participating);
            // DPRINT("logical_dfb_id: {}\n", logical_dfb_id);

            const uint32_t layout_byte_off = dfb_byte_offset_table[logical_dfb_id];
            // DPRINT("layout_byte_off: {}\n", layout_byte_off);
            volatile uint8_t* base_ptr = reinterpret_cast<volatile uint8_t*>(config_base + layout_byte_off);
            // DPRINT("base_ptr: 0x{:x}\n", static_cast<uint32_t>(reinterpret_cast<uintptr_t>(base_ptr)));
            volatile dfb_initializer_t* init_ptr = reinterpret_cast<volatile dfb_initializer_t*>(base_ptr);
            const uint16_t risc_mask = load_dfb_risc_mask(init_ptr);
            // DPRINT("risc_mask: 0x{:x}\n", risc_mask);

            volatile dfb_initializer_per_risc_t* per_risc_base = reinterpret_cast<volatile dfb_initializer_per_risc_t*>(
                config_base + layout_byte_off + sizeof(dfb_initializer_t));

            // --- Sub-pass A: populate g_dfb_interface ---
            {
                uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & risc_prefix_mask));
                volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
                ASSERT(compact_dfb_count < dfb::MAX_ACTIVE_DFBS_PACK);
                const uint8_t compact_dfb_id = compact_dfb_count++;
                g_dfb_logical_to_compact[logical_dfb_id] = compact_dfb_id;
                LocalDFBInterface& dfb_interface = g_dfb_interface[compact_dfb_id];
#else
                LocalDFBInterface& dfb_interface = g_dfb_interface[logical_dfb_id];
#endif

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

                // --- Sub-pass B (TRISC UCK_CHLKC_PACK only): TC hardware init merged into this loop ---
                // All producers (DM and TRISC) must wait for DM0's remapper enable before initializing TCs.
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
                if (load_dfb_per_risc_flags(per_risc_ptr) & 0x80u) {
                    while ((load_dfb_per_risc_flags(per_risc_ptr) & 0x40u) &&
                           !overlay::RemapperAPI::is_remapper_enabled());
                    for (int tc = 0; tc < load_dfb_num_tcs_to_rr(per_risc_ptr); tc++) {
                        dfb::PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                        uint8_t tc_id = dfb::get_counter_id(ptc);
                        ckernel::trisc::tile_counters[tc_id].f.reset = 1;
                        ckernel::trisc::tile_counters[tc_id].f.buf_capacity = init_ptr->capacity;
                    }
                    per_risc_ptr->num_tcs_and_init.tc_init_done = 1;
                }
#endif
            }

        }
    }

    // -----------------------------------------------------------------------
    // End merged loops (DM0 blob loop + shared per-DFB layout loop)
    // -----------------------------------------------------------------------

    // Point 4: all RISCs (DMs + TRISCs) capture merged loop end.
    // SW cost = t_after_merged_loop - start_time.
    t_after_merged_loop = rdcycle();

#ifndef COMPILE_FOR_TRISC
    // DM1 post-loop: burst-write accumulated remapper pair configs and enable the remapper.
    // Runs in parallel with DM0's ISR post-loop. Producers spin on is_remapper_enabled()
    // before TC init, so DM1's enable_remapper() is the gate for producer TC initialization.
    if (hartid == 1 && enable_remapper) {
        g_remapper_configurator.write_pairs_up_to(remapper_hwm);
        t_after_write_pairs_up_to = rdcycle();
        g_remapper_configurator.enable_remapper();
        end_remapper_config_time = rdcycle();
    }

    // DM0 post-loop: program ISR IE registers and enable/disable the DFB tile ISR.
    // The ISR setup is independent of the remapper; DM0 and DM1 post-loops run in parallel.
    if (hartid == 0) {
        uint64_t reg_val = CMDBUF_RD_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);
        reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(producer_txn_id_mask & 0xFFFFFFFFULL) << 32);
        CMDBUF_WR_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);

        reg_val = CMDBUF_RD_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);
        reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (consumer_txn_id_mask & 0xFFFFFFFFULL);
        CMDBUF_WR_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);
        t_after_isr_ie_writes = rdcycle();

        if ((producer_txn_id_mask | consumer_txn_id_mask) != 0) {
            enable_dfb_tile_isr();
            end_isr_enable_time = rdcycle();
        } else {
            disable_dfb_tile_isr();
        }
    }  // end DM0 post-loop

    // -----------------------------------------------------------------------
    // TC init loop (old Pass 3): all DMs; DM producers spin-wait for remapper.
    // DM0 writes implicit_sync_configured = 1 here once per DFB, after ISR setup is
    // complete, so other DMs/TRISCs can proceed with TC init.
    // Uses shared_base_ptr (shared per-DFB layout); no DM0 blob to skip.
    // -----------------------------------------------------------------------
    {
        uint32_t layout_cursor = per_dfb_layout_offset;
        for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
            volatile dfb_initializer_t* init_ptr =
                reinterpret_cast<volatile dfb_initializer_t*>(config_base + layout_cursor);
            const uint16_t risc_mask = load_dfb_risc_mask(init_ptr);
            uint8_t num_riscs = static_cast<uint8_t>(__builtin_popcount(risc_mask));

            volatile dfb_initializer_per_risc_t* per_risc_base = reinterpret_cast<volatile dfb_initializer_per_risc_t*>(
                config_base + layout_cursor + sizeof(dfb_initializer_t));

            // DM0 signals ISR-ready to all other DMs and TRISCs once per DFB, not on every poll iteration.
            if (hartid == 0) {
                init_ptr->implicit_sync_configured = 1;
            }

            if (risc_mask & hart_bit) {
                uint8_t risc_index = static_cast<uint8_t>(__builtin_popcount(risc_mask & risc_prefix_mask));
                volatile dfb_initializer_per_risc_t* per_risc_ptr = per_risc_base + risc_index;

                if (load_dfb_per_risc_flags(per_risc_ptr) & 0x80u) {
                    while ((load_dfb_per_risc_flags(per_risc_ptr) & 0x40u) &&
                           !overlay::RemapperAPI::is_remapper_enabled());
                    // Capture once (first producer DFB): marks end of spin-wait, start of TC HW writes.
                    // spinwait  = t_before_tc_writes - t_after_merged_loop
                    // tc_writes = t_after_tc_init_loop - t_before_tc_writes
                    if (!t_before_tc_writes) { t_before_tc_writes = rdcycle(); }
                    for (int tc = 0; tc < load_dfb_num_tcs_to_rr(per_risc_ptr); tc++) {
                        dfb::PackedTileCounter ptc = per_risc_ptr->packed_tile_counter[tc];
                        uint8_t tensix_id = dfb::get_tensix_id(ptc);
                        uint8_t tc_id = dfb::get_counter_id(ptc);
                        overlay::llk_intf_reset(tensix_id, tc_id);
                        overlay::llk_intf_set_capacity(tensix_id, tc_id, init_ptr->capacity);
                    }
                    per_risc_ptr->num_tcs_and_init.tc_init_done = 1;
                }
            }

            layout_cursor += static_cast<uint32_t>(
                sizeof(dfb_initializer_t) + (num_riscs * sizeof(dfb_initializer_per_risc_t)));
        }
    }
    // Point 6: isolates llk_intf_reset + llk_intf_set_capacity cost across all TCs/DFBs.
    // All DMs do TC init for their own producer TCs, so no hartid guard here.
    // Interval [end_isr_enable_time → t_after_tc_init_loop] = TC HW init cost per DM.
    // Interval [t_after_tc_init_loop → end_time] = wait_all_tcs_initialized spin time.
    t_after_tc_init_loop = rdcycle();
#endif

    wait_all_tcs_initialized(dfb_config_base, per_dfb_layout_offset, num_dfbs, hartid);
    uint32_t end_time = rdcycle();

    // All DPRINTs deferred to here so zero DPRINT overhead falls inside the timed region.
    DPRINT("start_time: {}\n", start_time);
#ifndef COMPILE_FOR_TRISC
    if (hartid == 0) {
        // DM0 per-DFB breakdown:
        //   subpassA   = loop overhead (header read, bulk-copy of ISR entry)
        //   subpassB_1 = HW cost: g_txn_dfb_descriptor populate + CMDBUF threshold register writes
        uint32_t total_subpassA  = 0;
        uint32_t total_subpassB1 = 0;
        for (uint32_t i = 0; i < num_dfbs; i++) {
            uint32_t prev_end   = (i == 0) ? start_time : t_subpassB[i - 1];
            uint32_t elapsed_a  = t_subpassA[i] - prev_end;
            uint32_t elapsed_b1 = t_subpassB[i] - t_subpassA[i];
            total_subpassA  += elapsed_a;
            total_subpassB1 += elapsed_b1;
            DPRINT("DFB{} subpassA={} subpassB_1={}\n", i, elapsed_a, elapsed_b1);
        }
        DPRINT("total_subpassA={} total_subpassB1={}\n", total_subpassA, total_subpassB1);

        // DM0 post-loop milestones.
        // [t_after_isr_ie_writes → end_isr_enable_time] = enable_dfb_tile_isr() cost
        if (t_after_isr_ie_writes) {
            DPRINT("t_after_isr_ie_writes: {}\n", t_after_isr_ie_writes);
        }
        if (end_isr_enable_time) {
            DPRINT("end_isr_enable_time: {}\n", end_isr_enable_time);
        }
    }
    if (hartid == 1) {
        // DM1 per-DFB breakdown: time to process remapper slots for each DFB.
        uint32_t total_rmp = 0;
        for (uint32_t i = 0; i < num_dfbs; i++) {
            uint32_t prev_end  = (i == 0) ? start_time : t_rmp_pass[i - 1];
            uint32_t elapsed   = t_rmp_pass[i] - prev_end;
            total_rmp += elapsed;
            DPRINT("DFB{} rmp_pass={}\n", i, elapsed);
        }
        DPRINT("total_rmp={}\n", total_rmp);

        // DM1 post-loop milestones.
        // [t_after_merged_loop → t_after_write_pairs_up_to] = write_pairs_up_to() HW burst
        // [t_after_write_pairs_up_to → end_remapper_config_time] = enable_remapper() cost
        if (t_after_write_pairs_up_to) {
            DPRINT("t_after_write_pairs_up_to: {}\n", t_after_write_pairs_up_to);
        }
        if (end_remapper_config_time) {
            DPRINT("end_remapper_config_time: {}\n", end_remapper_config_time);
        }
    }
    // All DMs: spinwait vs TC HW write split, then TC init loop end.
    if (t_before_tc_writes) {
        DPRINT(
            "spinwait={} tc_writes={}\n",
            t_before_tc_writes - t_after_merged_loop,
            t_after_tc_init_loop - t_before_tc_writes);
    }
    if (t_after_tc_init_loop) {
        DPRINT("t_after_tc_init_loop: {}\n", t_after_tc_init_loop);
    }
#endif
    // All RISCs (DMs + TRISCs): merged loop end and final end time.
    // SW cost  = t_after_merged_loop - start_time
    // wait cost= end_time - t_after_tc_init_loop  (DMs)
    //          = end_time - t_after_merged_loop    (TRISCs, which have t_after_tc_init_loop=0)
    if (t_after_merged_loop) {
        DPRINT("t_after_merged_loop: {}\n", t_after_merged_loop);
    }
    DPRINT("end_time: {}\n", end_time);
}
