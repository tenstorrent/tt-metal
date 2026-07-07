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
#include "internal/tt-2xx/quasar/dev_mem_map.h"
#include "internal/tt-2xx/risc_common.h"
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/llk_intf_api.hpp"
#else
#include "ckernel_trisc_common.h"
#endif

// Map a cached TL1 byte address to the uncached alias (DM↔TRISC visibility).
FORCE_INLINE volatile uint8_t* dfb_l1_uncached_byte_ptr(uintptr_t cached_l1_addr) {
    return reinterpret_cast<volatile uint8_t*>(
        static_cast<uintptr_t>(MEM_L1_UNCACHED_BASE) + (cached_l1_addr - static_cast<uintptr_t>(MEM_L1_BASE)));
}

FORCE_INLINE volatile uint32_t* dfb_l1_uncached_u32_ptr(uintptr_t cached_l1_addr) {
    return reinterpret_cast<volatile uint32_t*>(dfb_l1_uncached_byte_ptr(cached_l1_addr));
}

// Header fields shared across DM↔TRISC must be read via uncached alias on Quasar sim.
FORCE_INLINE uint32_t dfb_read_participation_mask(uintptr_t config_cached, uint8_t hart_u8) {
    return *dfb_l1_uncached_u32_ptr(
        config_cached + offsetof(dfb_global_header_t, participation_mask[0]) +
        static_cast<uintptr_t>(hart_u8) * sizeof(uint32_t));
}

FORCE_INLINE uint16_t dfb_read_hart_blob_offset(uintptr_t config_cached, uint8_t hart_u8) {
    return *reinterpret_cast<volatile uint16_t*>(dfb_l1_uncached_byte_ptr(
        config_cached + offsetof(dfb_global_header_t, hart_blob_offset[0]) +
        static_cast<uintptr_t>(hart_u8) * sizeof(uint16_t)));
}

// Init/wait blobs are host-written and polled across DM↔TRISC; read header bytes uncached.
FORCE_INLINE uint8_t dfb_read_blob_u8(uintptr_t blob_addr, uint32_t byte_off) {
    return *dfb_l1_uncached_byte_ptr(blob_addr + byte_off);
}

FORCE_INLINE uint32_t dfb_read_blob_u32(uintptr_t blob_addr, uint32_t byte_off) {
    return *dfb_l1_uncached_u32_ptr(blob_addr + byte_off);
}

// Device-side shadow of dfb_hart_init_entry_t, populated by dfb_read_init_entry_header.
// Mirrors the 24B on-disk layout (including txn_ids[4] + remapper_pair_index) captured in 6 u32 reads.
struct dfb_init_entry_hdr_t {
    uint8_t logical_dfb_id;
    uint8_t num_tcs;
    uint8_t flags;
    uint8_t capacity;
    uint32_t entry_size;
    uint32_t stride_size_precomp;  // host pre-computed: DM=raw bytes, TRISC=tile units (no device multiply needed)
    uint8_t stride_size_tiles;
    uint8_t num_txn_ids;
    uint8_t threshold;
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t producer_signal_bit;  // bit position in dfb_signal[dfb_id]; 0xFF if consumer
    uint8_t txn_ids[dfb::NUM_TXN_IDS];  // bytes 18–21 (DM only; 0 elsewhere)
    uint8_t broadcast_tc;         // DM pack byte 22 → iface.broadcast_tc (DM unpack only)
    uint8_t remapper_pair_index;  // byte 23 / DM pack byte 11 → iface._tc_align_pad on copy
};

// Fix A: read the entire 24B dfb_hart_init_entry_t (__attribute__((packed))) as 6 u32s.
// Two variants with identical unpack logic; only the pointer type differs:
//   - dfb_read_init_entry_header        — uncached alias (TRISC path, no L2 coherency)
//   - dfb_read_init_entry_header_cached — cached TL1 pointer (DM path, after L2 invalidate)
//
// Little-endian byte layout:
//   w0 [7:0]=logical_dfb_id  [15:8]=num_tcs  [23:16]=flags  [31:24]=capacity
//   w1 = entry_size (u32)
//   w2 = stride_size_precomp (u32): host pre-computed per hart type — DM=raw bytes, TRISC=tile units
//   w3 [7:0]=stride_size_tiles  [15:8]=num_txn_ids  [23:16]=threshold  [31:24]=num_entries_per_txn_id
//   w4 [7:0]=num_entries_per_txn_id_per_tc  [15:8]=producer_signal_bit  [23:16]=txn_ids[0]  [31:24]=txn_ids[1]
//   w5 [7:0]=txn_ids[2]  [15:8]=txn_ids[3]  [23:16]=remapper_pair_index  [31:24]=_pad

// Shared unpack helper: TRISC blob w3–w5 (legacy SoA byte layout).
template <typename PtrT>
FORCE_INLINE dfb_init_entry_hdr_t dfb_unpack_entry_header(PtrT s) {
    const uint32_t w0 = s[0], w1 = s[1], w2 = s[2], w3 = s[3], w4 = s[4], w5 = s[5];
    dfb_init_entry_hdr_t h;
    h.logical_dfb_id             = static_cast<uint8_t>(w0);
    h.num_tcs                    = static_cast<uint8_t>(w0 >> 8);
    h.flags                      = static_cast<uint8_t>(w0 >> 16);
    h.capacity                   = static_cast<uint8_t>(w0 >> 24);
    h.entry_size                 = w1;
    h.stride_size_precomp        = w2;
    h.stride_size_tiles          = static_cast<uint8_t>(w3);
    h.num_txn_ids                = static_cast<uint8_t>(w3 >> 8);
    h.threshold                  = static_cast<uint8_t>(w3 >> 16);
    h.num_entries_per_txn_id     = static_cast<uint8_t>(w3 >> 24);
    h.num_entries_per_txn_id_per_tc = static_cast<uint8_t>(w4);
    h.producer_signal_bit        = static_cast<uint8_t>(w4 >> 8);
    h.txn_ids[0]                 = static_cast<uint8_t>(w4 >> 16);
    h.txn_ids[1]                 = static_cast<uint8_t>(w4 >> 24);
    h.txn_ids[2]                 = static_cast<uint8_t>(w5);
    h.txn_ids[3]                 = static_cast<uint8_t>(w5 >> 8);
    h.broadcast_tc               = 0;
    h.remapper_pair_index        = static_cast<uint8_t>(w5 >> 16);
    return h;
}

// DM unpack: w3–w5 are the Opt-5 scalar pack (12B DTCM layout). See dfb_write_dm_scalar_pack_to_blob.
template <typename PtrT>
FORCE_INLINE dfb_init_entry_hdr_t dfb_unpack_entry_header_dm(PtrT s) {
    const uint32_t w0 = s[0], w1 = s[1], w2 = s[2], w3 = s[3], w4 = s[4], w5 = s[5];
    dfb_init_entry_hdr_t h;
    h.logical_dfb_id             = static_cast<uint8_t>(w0);
    h.num_tcs                    = static_cast<uint8_t>(w0 >> 8);
    h.flags                      = static_cast<uint8_t>(w0 >> 16);
    h.capacity                   = static_cast<uint8_t>(w0 >> 24);
    h.entry_size                 = w1;
    h.stride_size_precomp        = w2;
    h.stride_size_tiles          = 0;
    h.producer_signal_bit        = static_cast<uint8_t>(w3 >> 8);
    h.txn_ids[0]                 = static_cast<uint8_t>(w3 >> 16);
    h.txn_ids[1]                 = static_cast<uint8_t>(w3 >> 24);
    h.txn_ids[2]                 = static_cast<uint8_t>(w4);
    h.txn_ids[3]                 = static_cast<uint8_t>(w4 >> 8);
    h.threshold                  = static_cast<uint8_t>(w4 >> 16);
    h.num_entries_per_txn_id     = static_cast<uint8_t>(w4 >> 24);
    h.num_entries_per_txn_id_per_tc = static_cast<uint8_t>(w5);
    h.num_txn_ids                = static_cast<uint8_t>(w5 >> 8);
    h.broadcast_tc               = static_cast<uint8_t>(w5 >> 16);
    h.remapper_pair_index        = static_cast<uint8_t>(w5 >> 24);
    return h;
}

#ifndef COMPILE_FOR_TRISC
// Opt 4: write iface bytes [8,20) from unpacked eh — same 3×u32 layout as Opt 5 blob copy, no L1 reload.
FORCE_INLINE void dfb_write_dm_iface_scalars_from_hdr(LocalDFBInterface& iface, const dfb_init_entry_hdr_t& eh) {
    const uint32_t pack_w0 = static_cast<uint32_t>(eh.num_tcs) | (static_cast<uint32_t>(eh.txn_ids[0]) << 16) |
                             (static_cast<uint32_t>(eh.txn_ids[1]) << 24);
    const uint32_t pack_w1 = static_cast<uint32_t>(eh.txn_ids[2]) | (static_cast<uint32_t>(eh.txn_ids[3]) << 8) |
                             (static_cast<uint32_t>(eh.threshold) << 16) |
                             (static_cast<uint32_t>(eh.num_entries_per_txn_id) << 24);
    const uint32_t pack_w2 = static_cast<uint32_t>(eh.num_entries_per_txn_id_per_tc) |
                             (static_cast<uint32_t>(eh.num_txn_ids) << 8) |
                             (static_cast<uint32_t>(eh.broadcast_tc) << 16) |
                             (static_cast<uint32_t>(eh.remapper_pair_index) << 24);
    uint32_t* const dm_scalar = reinterpret_cast<uint32_t*>(&iface.num_tcs_to_rr);
    dm_scalar[0] = pack_w0;
    dm_scalar[1] = pack_w1;
    dm_scalar[2] = pack_w2;
    iface.tc_idx = 0;
}
#endif

// Uncached variant — used by TRISC (no private L2 cache).
FORCE_INLINE dfb_init_entry_hdr_t dfb_read_init_entry_header(uintptr_t entry_addr) {
    return dfb_unpack_entry_header(dfb_l1_uncached_u32_ptr(entry_addr));
}

// Cached variant — used by DM after invalidate_l2_cache_range covers the blob.
// Plain (non-volatile) pointer: no TL1 bypass; reads go through L1 D$ → L2 → TL1.
// Bytes [12,24) are in Opt-5 DTCM order (host writes via dfb_write_dm_scalar_pack_to_blob).
FORCE_INLINE dfb_init_entry_hdr_t dfb_read_init_entry_header_cached(uintptr_t entry_addr) {
    return dfb_unpack_entry_header_dm(reinterpret_cast<const uint32_t*>(entry_addr));
}

FORCE_INLINE volatile uint32_t* dfb_init_timing_slot_words(uint8_t slot) {
    // Device writes via uncached L1 alias; host reads the mirrored cached offset (TL1 @ 0x3ffc00).
    return reinterpret_cast<volatile uint32_t*>(
        static_cast<uintptr_t>(MEM_L1_UNCACHED_BASE) +
        static_cast<uintptr_t>(dfb::DFB_INIT_TIMING_L1_BYTE_OFFSET) +
        static_cast<uint32_t>(slot) * dfb::DFB_INIT_TIMING_WORDS_PER_SLOT * sizeof(uint32_t));
}

FORCE_INLINE void dfb_init_timing_write_slot(
    uint8_t slot,
    uint8_t role,
    uint32_t e2e,
    uint32_t metric_a,
    uint32_t metric_b,
    uint32_t metric_c,
    uint32_t metric_d,
    uint32_t metric_e,
    uint32_t metric_f,
    uint32_t start_time,
    uint32_t end_time,
    uint32_t metric_g = 0,
    uint32_t metric_h = 0,
    uint32_t metric_i = 0,
    uint32_t metric_j = 0) {
    volatile uint32_t* words = dfb_init_timing_slot_words(slot);
    // Publish pattern: write payload first, VALID last. Uncached stores land in TL1 directly.
    words[dfb::DFB_INIT_TIMING_W_MAGIC] = dfb::DFB_INIT_TIMING_MAGIC;
    words[dfb::DFB_INIT_TIMING_W_ROLE] = role;
    words[dfb::DFB_INIT_TIMING_W_E2E] = e2e;
    words[dfb::DFB_INIT_TIMING_W_METRIC_A] = metric_a;
    words[dfb::DFB_INIT_TIMING_W_METRIC_B] = metric_b;
    words[dfb::DFB_INIT_TIMING_W_METRIC_C] = metric_c;
    words[dfb::DFB_INIT_TIMING_W_METRIC_D] = metric_d;
    words[dfb::DFB_INIT_TIMING_W_METRIC_E] = metric_e;
    words[dfb::DFB_INIT_TIMING_W_METRIC_F] = metric_f;
    words[dfb::DFB_INIT_TIMING_W_START] = start_time;
    words[dfb::DFB_INIT_TIMING_W_END] = end_time;
    words[dfb::DFB_INIT_TIMING_W_METRIC_G] = metric_g;
    words[dfb::DFB_INIT_TIMING_W_METRIC_H] = metric_h;
    words[dfb::DFB_INIT_TIMING_W_METRIC_I] = metric_i;
    words[dfb::DFB_INIT_TIMING_W_METRIC_J] = metric_j;
    asm volatile("fence w, w" ::: "memory");
    words[dfb::DFB_INIT_TIMING_W_VALID] = 1u;
    asm volatile("fence w, w" ::: "memory");
}

#ifdef COMPILE_FOR_TRISC
FORCE_INLINE uint8_t dfb_init_timing_trisc_slot_index() {
    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
#if defined(UCK_CHLKC_PACK)
    return static_cast<uint8_t>(8u + neo_id * 2u + 1u);
#else
    return static_cast<uint8_t>(8u + neo_id * 2u);
#endif
}
#endif

inline uint32_t rdcycle() {
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c));
    return c;
}

// (load_dfb_risc_mask removed — device no longer walks the risc_mask in wait_all)


// Poll until all producers for DFB `dfb_id` have published their signal byte and
// (for DMs with implicit sync) DM0 has armed the ISR.
//
// Signal region layout at dfb_signal_region_off:
//   uint8_t  dfb_signal[NUM_DFBS * MAX_NUM_TILE_COUNTERS_TO_RR]  — per-producer byte slots (192B)
//   uint32_t dfb_expected_signal[NUM_DFBS]                        — host-computed bitmask (128B)
//
// Producer i of DFB d writes byte 1 to slot [d * MAX_PRODUCERS + i] (plain volatile, no AMO).
// Consumer reads dfb_expected_signal[d] (bitmask), then polls each set bit's byte slot.
//
// Runs for DMs unconditionally, and for TRISCs only when compiling for unpack or pack.
// Math (TRISC1) and TRISC3 compile this to an empty function so they never spin on
// uninitialised config data and never block DM0's wait_subordinates().
FORCE_INLINE void dfb_ensure_ready(uintptr_t config_cached, uint8_t dfb_id) {
#if defined(COMPILE_FOR_TRISC) && !defined(UCK_CHLKC_UNPACK) && !defined(UCK_CHLKC_PACK)
    return;
#endif
    const uint32_t sig_off = *dfb_l1_uncached_u32_ptr(
        config_cached + offsetof(dfb_global_header_t, dfb_signal_region_off));

    // dfb_expected_signal[dfb_id] sits after all producer byte slots.
    constexpr uint32_t kSlotStride = static_cast<uint32_t>(::dfb::MAX_NUM_TILE_COUNTERS_TO_RR);
    constexpr uint32_t kExpectedBase = static_cast<uint32_t>(::dfb::NUM_DFBS) * kSlotStride;
    const volatile uint32_t* exp_ptr = reinterpret_cast<const volatile uint32_t*>(
        dfb_l1_uncached_byte_ptr(
            config_cached + sig_off + kExpectedBase + static_cast<uint32_t>(dfb_id) * sizeof(uint32_t)));

    const uint32_t expected = *exp_ptr;
    if (expected == 0) {
        return;
    }

#ifndef COMPILE_FOR_TRISC
    const bool need_isr_gate =
        (*dfb_l1_uncached_byte_ptr(config_cached + offsetof(dfb_global_header_t, has_dm0_isr)) != 0);
    const volatile uint8_t* isr_ready_ptr =
        dfb_l1_uncached_byte_ptr(config_cached + offsetof(dfb_global_header_t, dm0_isr_ready));
#endif

    WAYPOINT("DFW");
    // Poll each producer's byte slot; remove from remaining when non-zero.
    uint32_t remaining = expected;
    while (remaining != 0u) {
        const uint8_t bit = static_cast<uint8_t>(__builtin_ctz(remaining));
        const volatile uint8_t* slot = dfb_l1_uncached_byte_ptr(
            config_cached + sig_off +
            static_cast<uint32_t>(dfb_id) * kSlotStride + bit);
        if (*slot != 0u) {
            remaining &= remaining - 1u;
        }
    }
#ifndef COMPILE_FOR_TRISC
    if (need_isr_gate) {
        while (*isr_ready_ptr != 1u) {}
    }
#endif
    WAYPOINT("DFD");
}

// Publish producer readiness: write byte 1 into this producer's unique slot in dfb_signal.
// producer_signal_bit == 0xFF means consumer (no signal slot) → no-op.
// dfb_signal_base must be the uncached alias of (config_base + dfb_signal_region_off),
// hoisted once before the init loop so per-entry publish is index + store only.
// A compiler barrier prevents the signal write from being hoisted before any preceding TC init.
FORCE_INLINE void dfb_publish_producer_ready(
    volatile uint8_t* dfb_signal_base, uint8_t logical_dfb_id, uint8_t producer_signal_bit) {
    if (producer_signal_bit == 0xFFu) {
        return;
    }
    constexpr uint32_t k_slot_stride = static_cast<uint32_t>(::dfb::MAX_NUM_TILE_COUNTERS_TO_RR);
    asm volatile("" ::: "memory");  // compiler barrier only; no hardware fence needed
    dfb_signal_base[static_cast<uint32_t>(logical_dfb_id) * k_slot_stride +
                    static_cast<uint32_t>(producer_signal_bit)] = 1u;
    WAYPOINT("PPR");
}

// Uncached-base + offset wrapper for one-off publishes outside the init loop.
FORCE_INLINE void dfb_publish_producer_ready(
    uintptr_t config_cached, uint32_t dfb_signal_region_off, uint8_t logical_dfb_id, uint8_t producer_signal_bit) {
    dfb_publish_producer_ready(
        dfb_l1_uncached_byte_ptr(config_cached + dfb_signal_region_off), logical_dfb_id, producer_signal_bit);
}

#ifndef COMPILE_FOR_TRISC

FORCE_INLINE void setup_dfb_implicit_sync(uint32_t tt_l1_ptr* dfb_config_base, uint32_t /*num_dfbs*/) {
    uint32_t start_time = rdcycle();

    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    uint32_t dm0_isr_blob_offset = ghdr->dm0_isr_blob_offset;

    volatile tt_l1_ptr uint8_t* dm0_blob_ptr = config_base + dm0_isr_blob_offset;

    uint32_t producer_txn_id_mask = 0;
    uint32_t consumer_txn_id_mask = 0;

    // Core-wide masks precomputed on host (Phase A).
    const volatile tt_l1_ptr uint32_t* core_hdr_src =
        reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(dm0_blob_ptr);
    producer_txn_id_mask = core_hdr_src[0];
    consumer_txn_id_mask = core_hdr_src[1];
    dm0_blob_ptr += sizeof(dfb_dm0_isr_blob_core_header_t);

    WAYPOINT("IS1");

    const uint32_t txn_hw_bytes =
        dm0_isr_txn_hw_pool_byte_size(producer_txn_id_mask, consumer_txn_id_mask);
    const uint32_t pool_bytes = dm0_isr_txn_desc_pool_byte_size(producer_txn_id_mask, consumer_txn_id_mask);
    const volatile tt_l1_ptr uint8_t* pool_base = dm0_blob_ptr + txn_hw_bytes;

    const volatile dfb_dm0_isr_txn_threshold_t* txn_threshold_table =
        reinterpret_cast<const volatile dfb_dm0_isr_txn_threshold_t*>(dm0_blob_ptr);

    const uint32_t t_before_desc_copy = rdcycle();
    if (pool_bytes != 0) {
        const uint32_t all_mask = producer_txn_id_mask | consumer_txn_id_mask;
        uint32_t pending_desc = all_mask;
        while (pending_desc) {
            const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending_desc));
            const volatile uint32_t* s = reinterpret_cast<const volatile uint32_t*>(pool_base + txn_id * 32u);
            volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(&g_txn_dfb_descriptor[txn_id]);
            d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
            d[4] = s[4]; d[5] = s[5]; d[6] = s[6]; d[7] = s[7];
            pending_desc &= (pending_desc - 1u);
        }
    }
    const uint32_t t_after_desc_copy = rdcycle();

    uint32_t total_l1_read = 0;
    uint32_t total_rocc_issue = 0;
    uint32_t hw_reg_writes = 0;
    uint32_t hw_reg_write_cycles = 0;
    const uint32_t t_before_cmdbuf = t_after_desc_copy;

    uint32_t pending = producer_txn_id_mask;
    while (pending) {
        const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending));
        const uint32_t t_slot_start = rdcycle();
        const uint32_t threshold = txn_threshold_table[txn_id].threshold;
        const uint32_t t_after_l1 = rdcycle();
        total_l1_read += t_after_l1 - t_slot_start;
        const uint32_t t_wr_start = rdcycle();
        CMDBUF_CLEAR_TILES_TO_PROCESS_TR_ACK(OVERLAY_RD_CMD_BUF, txn_id);
        asm volatile("nop");
        DPRINT("ack txn id: {} threshold: {}\n", txn_id, threshold);
        SET_TILES_TO_PROCESS_THRES_TR_ACK(txn_id, threshold);
        hw_reg_writes += 2u;
        hw_reg_write_cycles += rdcycle() - t_wr_start;
        const uint32_t t_after_rocc = rdcycle();
        total_rocc_issue += t_after_rocc - t_after_l1;
        pending &= (pending - 1u);
    }

    pending = consumer_txn_id_mask;
    while (pending) {
        const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending));
        const uint32_t t_slot_start = rdcycle();
        const uint32_t threshold = txn_threshold_table[txn_id].threshold;
        const uint32_t t_after_l1 = rdcycle();
        total_l1_read += t_after_l1 - t_slot_start;
        const uint32_t t_wr_start = rdcycle();
        CMDBUF_CLEAR_TILES_TO_PROCESS_WR_SENT(OVERLAY_WR_CMD_BUF, txn_id);
        asm volatile("nop");
        DPRINT("wr sent txn id: {} threshold: {}\n", txn_id, threshold);
        SET_TILES_TO_PROCESS_THRES_WR_SENT(txn_id, threshold);
        hw_reg_writes += 2u;
        hw_reg_write_cycles += rdcycle() - t_wr_start;
        const uint32_t t_after_rocc = rdcycle();
        total_rocc_issue += t_after_rocc - t_after_l1;
        pending &= (pending - 1u);
    }
    const uint32_t t_after_cmdbuf = rdcycle();

    const uint32_t t_before_ie = rdcycle();
    uint64_t reg_val = CMDBUF_RD_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(producer_txn_id_mask & 0xFFFFFFFFULL) << 32);
    DPRINT("producer txn id mask: {}\n", producer_txn_id_mask);
    CMDBUF_WR_REG(OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET, reg_val);
    const uint32_t t_after_first_ie_rmw = rdcycle();

    reg_val = CMDBUF_RD_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (consumer_txn_id_mask & 0xFFFFFFFFULL);
    DPRINT("consumer txn id mask: {}\n", consumer_txn_id_mask);
    CMDBUF_WR_REG(OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET, reg_val);
    const uint32_t t_after_isr_ie_writes = rdcycle();
    hw_reg_writes += 2u;
    hw_reg_write_cycles += t_after_isr_ie_writes - t_before_ie;

    if ((producer_txn_id_mask | consumer_txn_id_mask) != 0) {
        const uint32_t t_isr_start = rdcycle();
        enable_dfb_tile_isr();
        hw_reg_writes += 2u;  // mie + mstatus CSR writes
        hw_reg_write_cycles += rdcycle() - t_isr_start;
    } else {
        const uint32_t t_isr_start = rdcycle();
        disable_dfb_tile_isr();
        hw_reg_writes += 2u;
        hw_reg_write_cycles += rdcycle() - t_isr_start;
    }
    const uint32_t end_isr_enable_time = rdcycle();

    *dfb_l1_uncached_byte_ptr(
        reinterpret_cast<uintptr_t>(dfb_config_base) + offsetof(dfb_global_header_t, dm0_isr_ready)) = 1u;
    asm volatile("fence w, w" ::: "memory");

    WAYPOINT("ISD");

    const uint32_t end_time = rdcycle();

    const uint32_t pre_loop_sw = t_before_desc_copy - start_time;
    const uint32_t subpassB_desc = t_after_desc_copy - t_before_desc_copy;
    const uint32_t subpassB_hw = t_after_cmdbuf - t_before_cmdbuf;
    const uint32_t first_ie_rmw = t_after_first_ie_rmw - t_before_ie;
    const uint32_t second_ie_rmw = t_after_isr_ie_writes - t_after_first_ie_rmw;
    const uint32_t isr_enable = (end_isr_enable_time > t_after_isr_ie_writes)
                                    ? end_isr_enable_time - t_after_isr_ie_writes
                                    : 0;
    dfb_init_timing_write_slot(
        0,
        dfb::DFB_INIT_TIMING_ROLE_DM0_ISR,
        end_time - start_time,
        pre_loop_sw,
        subpassB_desc,
        hw_reg_write_cycles,
        total_l1_read,
        total_rocc_issue,
        first_ie_rmw,
        start_time,
        end_time,
        second_ie_rmw,
        isr_enable,
        hw_reg_writes,
        subpassB_hw);
}

FORCE_INLINE void setup_dfb_remapper(uint32_t tt_l1_ptr* dfb_config_base, uint32_t num_dfbs) {
    const uint32_t start_time = rdcycle();

    volatile tt_l1_ptr uint8_t* config_base = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dfb_config_base);
    volatile dfb_global_header_t* ghdr = reinterpret_cast<volatile dfb_global_header_t*>(config_base);
    num_dfbs = ghdr->num_dfbs;
    const uint32_t dm1_remapper_blob_offset = ghdr->dm1_remapper_blob_offset;
    volatile tt_l1_ptr uint8_t* dm1_blob_ptr = config_base + dm1_remapper_blob_offset;

    // bool enable_remapper = false;
    uint32_t end_remapper_config_time = 0;

    WAYPOINT("RS");

    uint32_t blob_l1_read_sw = 0;
    uint32_t pairs_reg_hw = 0;
    uint32_t blob_loop_ovhd = 0;
    uint32_t pairs_slots_written = 0;
    uint32_t first_pair_clientR_hw = 0;
    uint32_t first_pair_clientL_hw = 0;
    uint32_t last_pair_hw = 0;
    bool first_slot_written = false;
    uint32_t max_pair_idx = 0;

    for (uint32_t logical_dfb_id = 0; logical_dfb_id < num_dfbs; logical_dfb_id++) {
        const uint32_t t_pass_start = rdcycle();
        uint32_t pass_l1_read = 0;
        uint32_t pass_pairs_reg = 0;

        const volatile dfb_dm1_remapper_entry_header_t* entry_hdr =
            reinterpret_cast<const volatile dfb_dm1_remapper_entry_header_t*>(dm1_blob_ptr);
        const int num_rmp = entry_hdr->num_remapper_slots;
        WAYPOINT("RS2");

        const uint32_t entry_bytes = sizeof(dfb_dm1_remapper_entry_header_t)
                                     + static_cast<uint32_t>(num_rmp) * sizeof(dfb_dm0_remapper_slot_t);
        const volatile dfb_dm0_remapper_slot_t* slots =
            reinterpret_cast<const volatile dfb_dm0_remapper_slot_t*>(
                dm1_blob_ptr + sizeof(dfb_dm1_remapper_entry_header_t));
        const uint32_t t_after_hdr = rdcycle();
        pass_l1_read += t_after_hdr - t_pass_start;

        for (int s = 0; s < num_rmp; s++) {
            const uint32_t t_slot_read_start = rdcycle();
            const uint32_t pair_idx = slots[s].pair_index;
            const uint32_t clientR_val = slots[s].clientR_val;
            const uint32_t clientL_val = slots[s].clientL_val;
            const uint32_t t_after_slot_read = rdcycle();
            pass_l1_read += t_after_slot_read - t_slot_read_start;

            const uint32_t t_clientR_start = rdcycle();
            WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pair_idx), clientR_val);
            const uint32_t t_after_clientR = rdcycle();
            if (!first_slot_written) {
                first_pair_clientR_hw = t_after_clientR - t_clientR_start;
            }

            const uint32_t t_clientL_start = rdcycle();
            WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx), clientL_val);
            const uint32_t t_after_clientL = rdcycle();
            if (!first_slot_written) {
                first_pair_clientL_hw = t_after_clientL - t_clientL_start;
                first_slot_written = true;
            }

            const uint32_t pair_reg_hw = (t_after_clientR - t_clientR_start) + (t_after_clientL - t_clientL_start);
            pass_pairs_reg += pair_reg_hw;
            last_pair_hw = pair_reg_hw;
            pairs_slots_written++;
            if (pair_idx > max_pair_idx) {
                max_pair_idx = pair_idx;
            }
        }

        dm1_blob_ptr += entry_bytes;
        const uint32_t t_pass_end = rdcycle();
        blob_l1_read_sw += pass_l1_read;
        pairs_reg_hw += pass_pairs_reg;
        blob_loop_ovhd += (t_pass_end - t_pass_start) - pass_l1_read - pass_pairs_reg;
    }

    // uint32_t enable_remapper_hw = 0;
    // if (enable_remapper) {
    //     const uint32_t t_before_enable = rdcycle();
    //     g_remapper_configurator.enable_remapper();
    //     end_remapper_config_time = rdcycle();
    //     enable_remapper_hw = end_remapper_config_time - t_before_enable;
    // }

    if (pairs_slots_written > 0u) {
        g_remapper_configurator.set_pair_high_watermark(max_pair_idx);
    }

    WAYPOINT("RSD");
    const uint32_t end_time = rdcycle();

    dfb_init_timing_write_slot(
        1,
        dfb::DFB_INIT_TIMING_ROLE_DM1_RMP,
        end_time - start_time,
        blob_l1_read_sw,
        blob_loop_ovhd,
        pairs_reg_hw,
        0,
        first_pair_clientR_hw,
        first_pair_clientL_hw,
        start_time,
        end_time,
        last_pair_hw,
        pairs_reg_hw,                        // METRIC_H: hw_reg_write_cycles
        pairs_slots_written * 2u,            // METRIC_I: hw_reg_writes (clientR + clientL per slot)
        pairs_slots_written);
}

#endif  // !COMPILE_FOR_TRISC

// DM0/DM1 coordinators run setup_dfb_implicit_sync / setup_dfb_remapper from DM firmware (no TC wait).
// DM2-7 + TRISC: walk this hart's pre-computed sequential init blob; TC readiness is published
// via atomic OR into dfb_signal[dfb_id] and consumers poll in DataflowBuffer::DataflowBuffer().
FORCE_INLINE void setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t /*local_dfb_mask*/) {
    const uint32_t start_time = rdcycle();

#ifdef COMPILE_FOR_TRISC
    const uint32_t neo_id = ckernel::csr_read<ckernel::CSR::NEO_ID>();
    const uint8_t hart_u8 = static_cast<uint8_t>(8u + neo_id);
#else
    uint64_t hartid_raw;
    asm volatile("csrr %0, mhartid" : "=r"(hartid_raw));
    const uint8_t hart_u8 = static_cast<uint8_t>(hartid_raw);
#endif

    const uintptr_t config_cached = reinterpret_cast<uintptr_t>(dfb_config_base);
    g_dfb_config_base_addr = config_cached;

    const uint32_t participation_mask = dfb_read_participation_mask(config_cached, hart_u8);
    const uint32_t dfb_signal_region_off = *dfb_l1_uncached_u32_ptr(
        config_cached + offsetof(dfb_global_header_t, dfb_signal_region_off));
    // Hoist uncached signal-region base; per-entry publish adds dfb_id×stride + signal_bit only.
    volatile uint8_t* const dfb_signal_base =
        dfb_l1_uncached_byte_ptr(config_cached + dfb_signal_region_off);

    // One load: this hart's blob offset from the header — no dependent table arithmetic.
    const volatile uint8_t* p = reinterpret_cast<const volatile uint8_t*>(
        config_cached + dfb_read_hart_blob_offset(config_cached, hart_u8));

    uint32_t total_remapper_spin = 0;
    uint32_t total_tc_hw = 0;
    uint32_t total_tc_reset_hw = 0;
    uint32_t total_tc_capacity_hw = 0;
    uint32_t total_hw_reg_writes = 0;
    // Sub-breakdown metrics (g–j):
    //   g: pre-loop fixed overhead (3 uncached header reads before the loop)
    //   h: time in dfb_read_init_entry_header (6 u32 reads × N entries)
    //   i: time in TC-slot read+write loop (all entries)
    //   j: sig_slot address compute + uncached store (producers only)
    uint32_t total_entry_hdr = 0;
    uint32_t total_tc_slots  = 0;
    uint32_t total_sig_write = 0;

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    uint8_t compact_dfb_count = 0;
#endif

    // g: capture time before the loop (3 uncached header reads above + blob-offset load)
    const uint32_t t_loop_start = rdcycle();
    const uint32_t pre_loop = t_loop_start - start_time;

    // -----------------------------------------------------------------------
    // Sequential init blob walk: no pointer-table lookups, no per-DFB indirection.
    // Entry count = popcount(participation_mask[hart_u8]); blob starts at init entries.
    // Fix A: dfb_read_init_entry_header issues 6 u32 loads per entry (vs 12+ individual
    // byte/word reads in the baseline), and TC arrays are read as u32 pairs.
    //
    // Fix B (this loop):
    //   - LUT replaces dfb_hart_init_entry_byte_size (MUL×9 + align) for pointer advance
    //   - txn_ids copy unrolled to 4 unconditional stores (eliminates variable-bound loop)
    //   - tc_base/limit/packed_tc addresses computed with SHL+ADD (replaces separate muls)
    //
    // Fix C (DM only): one invalidate_l2_cache_range before the loop; all blob reads below
    //   use cached TL1 pointers so the DM's L1 D$ + L2 absorb the per-word latency.
    // -----------------------------------------------------------------------

    // Replaces dfb_hart_init_entry_byte_size for num_tcs 0..6: avoids MUL per entry.
    // AoP tail: 24B header + (num_tcs*9 + 3)&~3. Identical to original SoA byte count.
    static constexpr uint8_t k_entry_byte_size_lut[7] = {24, 36, 44, 52, 60, 72, 80};

    const uint8_t num_init = dfb_hart_participation_count(participation_mask);

    // Fix C (DM only): invalidate L2 cache lines covering this hart's init blob so that
    // the blob walk below reads through the DM's L1 D$ + L2 cache (amortized ~3 cyc/word
    // after line fills) rather than bypassing the cache entirely via the uncached alias
    // (~15 cyc/word per load). The global header fields read above (participation_mask,
    // dfb_signal_region_off, hart_blob_offset) and the signal region stay on the uncached
    // alias because they require cross-hart visibility without explicit cache management.
    //
    // Invalidate upper bound: num_init × 80B (max entry for 6 TCs) is a conservative
    // per-hart blob size. Over-estimates by at most 1 extra cache line for small num_tcs.
    // The invalidate discards any stale L2 lines so the first cached read fetches the
    // host-written TL1 bytes.
#ifndef COMPILE_FOR_TRISC
    if (num_init > 0) {
        const uintptr_t blob_start = reinterpret_cast<uintptr_t>(p);
        const uintptr_t blob_end   = blob_start + static_cast<uintptr_t>(num_init) * 80u;
        invalidate_l2_cache_range(blob_start, blob_end - blob_start);
    }

    // Opt 3: compute the starting g_dfb_interface pointer from the first blob entry's
    // logical_dfb_id (1 multiply, amortised). Subsequent iterations use pointer increment
    // (1 addi) instead of recomputing &g_dfb_interface[id] (id×140 multiply) per entry.
    // Precondition: host emits blob entries in ascending consecutive DFB ID order.
    LocalDFBInterface* iface_ptr = (num_init > 0)
        ? &g_dfb_interface[*reinterpret_cast<const uint8_t*>(reinterpret_cast<uintptr_t>(p))]
        : nullptr;
#endif

    for (uint8_t i = 0; i < num_init; i++) {
        const uintptr_t e_addr = reinterpret_cast<uintptr_t>(p);

        const uint32_t t_hdr_start = rdcycle();
#ifdef COMPILE_FOR_TRISC
        const dfb_init_entry_hdr_t eh = dfb_read_init_entry_header(e_addr);
#else
        const dfb_init_entry_hdr_t eh = dfb_read_init_entry_header_cached(e_addr);
#endif
        total_entry_hdr += rdcycle() - t_hdr_start;

        const uint8_t num_tcs = eh.num_tcs;

        // LUT pointer advance: 1 lbu + 1 add instead of MUL×9 + alignment mask.
        const uint32_t entry_bytes = (num_tcs < 7u)
            ? k_entry_byte_size_lut[num_tcs]
            : dfb_hart_init_entry_byte_size(num_tcs);
        p = reinterpret_cast<const volatile uint8_t*>(e_addr + entry_bytes);

        // AoP TC tail starts right after the 24B header: pairs first, ptc bytes after all pairs.
        const uintptr_t tc_base_addr = e_addr + sizeof(dfb_hart_init_entry_t);

        WAYPOINT("L1");

        // --- Resolve g_dfb_interface slot ---
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
        ASSERT(compact_dfb_count < dfb::MAX_ACTIVE_DFBS_PACK);
        const uint8_t compact_id = compact_dfb_count++;
        g_dfb_logical_to_compact[eh.logical_dfb_id] = compact_id;
        LocalDFBInterface& iface = g_dfb_interface[compact_id];
#elif defined(COMPILE_FOR_TRISC)
        LocalDFBInterface& iface = g_dfb_interface[eh.logical_dfb_id];
#else
        // Opt 3: advance running pointer (addi, not mul) — see initialisation above.
        LocalDFBInterface& iface = *iface_ptr++;
#endif

        // --- Common scalar fields ---
#if defined(COMPILE_FOR_TRISC)
        iface.num_tcs_to_rr = num_tcs;
        iface.tc_idx = 0;
#endif

        WAYPOINT("L2");

        // --- Role-specific scalar fields ---
#ifdef COMPILE_FOR_TRISC
        iface.entry_size        = static_cast<uint16_t>(eh.entry_size >> cb_addr_shift);
        // Opt 2: stride_size_precomp already has (entry_size >> shift) * stride_in_entries — no multiply.
        iface.stride_size       = static_cast<uint16_t>(eh.stride_size_precomp);
        iface.stride_size_tiles = eh.stride_size_tiles;
#if defined(UCK_CHLKC_PACK)
        iface.wr_entry_ptr = 0;
#else  // unpack TRISC
        iface.tensix_trisc_mask = static_cast<uint8_t>(eh.flags & DFB_HART_FLAG_TRISC_MASK);
#endif
#else  // DM
        iface.entry_size  = eh.entry_size >> cb_addr_shift;
        // Opt 2: stride_size_precomp already has entry_size_raw * stride_in_entries — no multiply.
        iface.stride_size = eh.stride_size_precomp;
        // Opt 4: 3×u32 DTCM stores from Opt-5 unpacked eh; no second L1 read of bytes 12–23.
        dfb_write_dm_iface_scalars_from_hdr(iface, eh);
#endif

        WAYPOINT("L3");

        // --- TC slot population ---
        // AoP TC tail: dfb_blob_tc_pair_t[num_tcs] (8B each, base+limit adjacent per slot)
        // immediately after the 24B header, then uint8_t ptc[num_tcs] padded to 4B.
        // Preload ptc as 1–2 word reads before the loop; in-loop ptc is a register bit-extract
        // (zero additional memory traffic). Running pair pointer advances 8B per slot.
        // Fix C: DM path uses cached pointer (blob in L2 after invalidate); TRISC uses uncached alias.
        const uintptr_t ptc_base_addr = tc_base_addr + (static_cast<uintptr_t>(num_tcs) << 3u);
#ifdef COMPILE_FOR_TRISC
        const volatile dfb_blob_tc_pair_t* pairs = reinterpret_cast<const volatile dfb_blob_tc_pair_t*>(
            dfb_l1_uncached_u32_ptr(tc_base_addr));
        const uint32_t ptc_w0 = (num_tcs > 0u) ? *dfb_l1_uncached_u32_ptr(ptc_base_addr) : 0u;
        const uint32_t ptc_w1 = (num_tcs > 4u) ? *dfb_l1_uncached_u32_ptr(ptc_base_addr + 4u) : 0u;
#else
        const dfb_blob_tc_pair_t* pairs = reinterpret_cast<const dfb_blob_tc_pair_t*>(tc_base_addr);
        const uint32_t ptc_w0 = (num_tcs > 0u) ? *reinterpret_cast<const uint32_t*>(ptc_base_addr) : 0u;
        const uint32_t ptc_w1 = (num_tcs > 4u) ? *reinterpret_cast<const uint32_t*>(ptc_base_addr + 4u) : 0u;
#endif
        const uint32_t t_slots_start = rdcycle();
        for (uint8_t t = 0; t < num_tcs; t++, pairs++) {
            // Host pre-shifts TRISC TC addresses to tile units at serialization; DM blobs keep byte addresses.
            const uint32_t base      = pairs->base_addr;
            const uint32_t limit     = pairs->limit;
            const uint8_t  ptc       = static_cast<uint8_t>((t < 4u ? ptc_w0 : ptc_w1) >> ((t & 3u) * 8u));
            iface.tc_slots[t].packed_tile_counter = ptc;

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
            iface.tc_slots[t].base_addr      = base;
            iface.tc_slots[t].wr_offset      = 0;
            iface.tc_slots[t].ring_size      = static_cast<uint16_t>(limit - base);
            iface.tc_slots[t].base_entry_idx = static_cast<uint16_t>(
                (base - iface.tc_slots[0].base_addr) / iface.entry_size);
            iface.tc_slots[t].wr_entry_idx   = iface.tc_slots[t].base_entry_idx;
#elif defined(COMPILE_FOR_TRISC)  // unpack
            iface.tc_slots[t].base_addr      = base;
            iface.tc_slots[t].rd_offset      = 0;
            iface.tc_slots[t].ring_size      = static_cast<uint16_t>(limit - base);
            iface.tc_slots[t].base_entry_idx = static_cast<uint16_t>(
                (base - iface.tc_slots[0].base_addr) / iface.entry_size);
            iface.tc_slots[t].rd_entry_idx   = iface.tc_slots[t].base_entry_idx;
#else  // DM
            iface.tc_slots[t].base_addr = base;
            iface.tc_slots[t].limit     = limit;
            iface.tc_slots[t].rd_ptr    = base;
            iface.tc_slots[t].wr_ptr    = base;
#endif
        }
        total_tc_slots += rdcycle() - t_slots_start;

        WAYPOINT("L4");

        // --- Producer-only: wait for remapper pair + TC HW init + publish ready ---
#if !defined(COMPILE_FOR_TRISC) || defined(UCK_CHLKC_PACK)
        if (eh.flags & DFB_HART_FLAG_IS_PRODUCER) {
            // Remapped producers: spin until DM1 has written this pair's ClientL config with
            // non-zero valid bits (bits [11:8]). No global remapper-enable wait is needed.
            if (eh.flags & DFB_HART_FLAG_REMAPPER_EN) {
                const uint32_t spin_start = rdcycle();
                const uint32_t pair_idx = eh.remapper_pair_index;
                WAYPOINT("RMSW");
                while (overlay::RemapperAPI::get_clientL_valid_hw(pair_idx) == 0u) {
                }
                WAYPOINT("RMSD");
                total_remapper_spin += rdcycle() - spin_start;
            }

            const uint32_t tc_hw_start = rdcycle();
            for (uint8_t t = 0; t < num_tcs; t++) {
                const uint8_t packed_ptc = iface.tc_slots[t].packed_tile_counter;
                const uint8_t tc_id = dfb::get_counter_id(packed_ptc);
#ifndef COMPILE_FOR_TRISC
                const uint8_t tensix_id = dfb::get_tensix_id(packed_ptc);
                const uint32_t t_reset = rdcycle();
                overlay::fast_llk_intf_reset(tensix_id, tc_id);
                total_tc_reset_hw += rdcycle() - t_reset;
                total_hw_reg_writes++;
                const uint32_t t_cap = rdcycle();
                overlay::fast_llk_intf_set_capacity(tensix_id, tc_id, eh.capacity);
                total_tc_capacity_hw += rdcycle() - t_cap;
                total_hw_reg_writes++;
#elif defined(UCK_CHLKC_PACK)
                const uint32_t t_reset = rdcycle();
                ckernel::trisc::tile_counters[tc_id].f.reset = 1;
                total_tc_reset_hw += rdcycle() - t_reset;
                total_hw_reg_writes++;
                const uint32_t t_cap = rdcycle();
                ckernel::trisc::tile_counters[tc_id].f.buf_capacity = eh.capacity;
                total_tc_capacity_hw += rdcycle() - t_cap;
                total_hw_reg_writes++;
#endif
            }
            total_tc_hw += rdcycle() - tc_hw_start;

            const uint32_t t_sig_start = rdcycle();
            dfb_publish_producer_ready(dfb_signal_base, eh.logical_dfb_id, eh.producer_signal_bit);
            total_sig_write += rdcycle() - t_sig_start;
        }
#endif  // !COMPILE_FOR_TRISC || UCK_CHLKC_PACK

        WAYPOINT("L5");
    }

    const uint32_t t_after_merged_loop = rdcycle();

    WAYPOINT("L12");
    const uint32_t end_time = rdcycle();

#ifdef COMPILE_FOR_TRISC
    const uint8_t timing_slot = dfb_init_timing_trisc_slot_index();
    const uint8_t timing_role = dfb::DFB_INIT_TIMING_ROLE_TRISC_LOCAL;
#else
    const uint8_t timing_slot = hart_u8;
    const uint8_t timing_role = dfb::DFB_INIT_TIMING_ROLE_DM_LOCAL;
#endif
    dfb_init_timing_write_slot(
        timing_slot,
        timing_role,
        end_time - start_time,
        t_after_merged_loop - start_time,   // METRIC_A: merged_sw (full loop wall time)
        total_remapper_spin,                 // METRIC_B: remapper_spin
        total_tc_hw,                         // METRIC_C: tc_hw
        total_hw_reg_writes,                 // METRIC_D: hw_reg_writes (TC reset + capacity per producer TC)
        total_tc_reset_hw,                   // METRIC_E: tc_reset_hw
        total_tc_capacity_hw,                // METRIC_F: tc_capacity_hw
        start_time,
        end_time,
        pre_loop,                            // METRIC_G: pre_loop overhead (3 uncached header loads)
        total_entry_hdr,                     // METRIC_H: entry_hdr (6 u32 bulk reads × N entries)
        total_tc_slots,                      // METRIC_I: tc_slots (base/limit/ptc reads + iface writes)
        total_sig_write);                    // METRIC_J: sig_write (uncached store to signal region)
}



// 1k cycles to program isrs in the worst case
// programming remapper is 37 cycles / remapper pair (how many in worst case??)
// enabling remapper is 4-100 cycles (why the range?)
// tc reset + set cap is 45 cycles / tc (worst case how many???)
