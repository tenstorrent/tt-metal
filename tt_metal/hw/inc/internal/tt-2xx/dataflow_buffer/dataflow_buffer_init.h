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

// The region offsets in dfb_global_header_t are uint16_t; reading one with the u32 accessor would
// fold the two following bytes into the high half.
FORCE_INLINE volatile uint16_t* dfb_l1_uncached_u16_ptr(uintptr_t cached_l1_addr) {
    return reinterpret_cast<volatile uint16_t*>(dfb_l1_uncached_byte_ptr(cached_l1_addr));
}

// Header fields shared across DM↔TRISC must be read via uncached alias on Quasar sim.
FORCE_INLINE uint64_t dfb_read_hart_desc(uintptr_t config_cached, uint8_t hart_u8) {
    return *reinterpret_cast<volatile uint64_t*>(dfb_l1_uncached_byte_ptr(
        config_cached + offsetof(dfb_global_header_t, hart_desc[0]) +
        static_cast<uintptr_t>(hart_u8) * sizeof(dfb_hart_desc_t)));
}

// Device-side shadow of dfb_hart_init_entry_t (TRISC only), populated by dfb_read_init_entry_header.
struct dfb_init_entry_hdr_t {
    uint32_t entry_size;
    uint32_t stride_size_precomp;  // host pre-computed, tile units
    uint16_t num_entries;
    uint16_t capacity;
    uint8_t logical_dfb_id;
    uint8_t num_tcs;
    uint8_t flags;
    uint8_t stride_size_tiles;
    uint8_t num_txn_ids;
    uint8_t threshold;
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t producer_signal_bit;  // bit position in dfb_signal[dfb_id]; 0xFF if consumer
    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t broadcast_tc;         // unused on TRISC; kept so the shadow stays 36B
    uint8_t remapper_pair_index;  // byte 22
    uint8_t intra_shadow_tc_id;   // byte 23: intra-tensix ClientR shadow TC; 0xFF if unused
    uint16_t block_size;          // bytes 28-29: how many tiles exist in one block
    uint8_t split_tc;             // byte 3: tiles are split across all tile counters (1) or not (0)
    uint16_t entries_to_jump;     // bytes 30-31: tiles to jump on each rotation of tile counters
};
static_assert(sizeof(dfb_init_entry_hdr_t) == 36, "dfb_init_entry_hdr_t must pack to 36B with no padding");
static_assert(alignof(dfb_init_entry_hdr_t) == 4, "dfb_init_entry_hdr_t alignment should follow uint32_t");

// Read the entire 32B dfb_hart_init_entry_t (__attribute__((packed))) as 8 u32s, through the
// uncached alias (TRISC path; DM entries are a host-built interface image instead).
//
// Little-endian byte layout:
//   w0 [7:0]=logical_dfb_id  [15:8]=num_tcs  [23:16]=flags  [31:24]=split_tc
//   w1 = entry_size (u32)
//   w2 = stride_size_precomp (u32): host pre-computed, tile units
//   w3 [7:0]=stride_size_tiles  [15:8]=num_txn_ids  [23:16]=threshold  [31:24]=num_entries_per_txn_id
//   w4 [7:0]=num_entries_per_txn_id_per_tc  [15:8]=producer_signal_bit  [23:16]=txn_ids[0]  [31:24]=txn_ids[1]
//   w5 [7:0]=txn_ids[2]  [15:8]=txn_ids[3]  [23:16]=remapper_pair_index
//      [31:24]=intra_shadow_tc_id
//   w6 [15:0]=num_entries  [31:16]=capacity
//   w7 [15:0]=block_size  [31:16]=entries_to_jump (u16)

template <typename PtrT>
FORCE_INLINE dfb_init_entry_hdr_t dfb_unpack_entry_header(PtrT s) {
    const uint32_t w0 = s[0], w1 = s[1], w2 = s[2], w3 = s[3], w4 = s[4], w5 = s[5], w6 = s[6], w7 = s[7];
    dfb_init_entry_hdr_t h;
    h.logical_dfb_id = static_cast<uint8_t>(w0);
    h.num_tcs = static_cast<uint8_t>(w0 >> 8);
    h.flags = static_cast<uint8_t>(w0 >> 16);
    h.split_tc = static_cast<uint8_t>(w0 >> 24);
    h.entry_size = w1;
    h.stride_size_precomp = w2;
    h.stride_size_tiles = static_cast<uint8_t>(w3);
    h.num_txn_ids = static_cast<uint8_t>(w3 >> 8);
    h.threshold = static_cast<uint8_t>(w3 >> 16);
    h.num_entries_per_txn_id = static_cast<uint8_t>(w3 >> 24);
    h.num_entries_per_txn_id_per_tc = static_cast<uint8_t>(w4);
    h.producer_signal_bit = static_cast<uint8_t>(w4 >> 8);
    h.txn_ids[0] = static_cast<uint8_t>(w4 >> 16);
    h.txn_ids[1] = static_cast<uint8_t>(w4 >> 24);
    h.txn_ids[2] = static_cast<uint8_t>(w5);
    h.txn_ids[3] = static_cast<uint8_t>(w5 >> 8);
    h.broadcast_tc = 0;
    h.remapper_pair_index = static_cast<uint8_t>(w5 >> 16);
    h.intra_shadow_tc_id = static_cast<uint8_t>(w5 >> 24);
    h.num_entries = static_cast<uint16_t>(w6);
    h.capacity = static_cast<uint16_t>(w6 >> 16);
    h.block_size = static_cast<uint16_t>(w7);
    h.entries_to_jump = static_cast<uint16_t>(w7 >> 16);
    return h;
}

// Uncached reader -- TRISC only; DM entries are a host-built interface image instead.
FORCE_INLINE dfb_init_entry_hdr_t dfb_read_init_entry_header(uintptr_t entry_addr) {
    return dfb_unpack_entry_header(dfb_l1_uncached_u32_ptr(entry_addr));
}
FORCE_INLINE volatile uint32_t* dfb_init_timing_slot_words(uint8_t slot) {
    // Device writes via uncached L1 alias; host reads the mirrored cached offset (TL1 @ 0x3ffc00).
    return reinterpret_cast<volatile uint32_t*>(
        static_cast<uintptr_t>(MEM_L1_UNCACHED_BASE) + static_cast<uintptr_t>(dfb::DFB_INIT_TIMING_L1_BYTE_OFFSET) +
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
#ifdef DFB_INIT_TIMING_ENABLED
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
#else
    (void)slot;
    (void)role;
    (void)e2e;
    (void)metric_a;
    (void)metric_b;
    (void)metric_c;
    (void)metric_d;
    (void)metric_e;
    (void)metric_f;
    (void)start_time;
    (void)end_time;
    (void)metric_g;
    (void)metric_h;
    (void)metric_i;
    (void)metric_j;
#endif
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

// Whole-function start/end timestamps. Kept outside the call boundary's window by the noinline
// wrappers in dm.cc: a "memory" clobber does not stop the caller's CSR writes and volatile accesses
// from being scheduled into an inlined window, only a call boundary does.
inline uint32_t rdcycle_e2e() {
#ifdef DFB_INIT_TIMING_ENABLED
    uint32_t c;
    asm volatile("rdcycle %0" : "=r"(c)::"memory");
    return c;
#else
    return 0;
#endif
}

// Pause between a producer's polls of its remapper pair's ClientL config register. A tight poll is
// a back-to-back uncached read of the same remapper register block DM1 is writing, and four
// producers doing that at once more than doubled DM1's remapper setup. Worst hart median over 24
// cold iterations of a four-producer case:
//   no pause 3513 | 64 cycles 2196 | 100 cycles 2132 | 256 cycles 2063
// A longer pause costs a producer up to that many cycles of late wake-up once DM1 has written its
// pair. uint32_t because rdcycle is a single 32-bit register on the RV32 TRISCs, which compile this
// loop too.
constexpr uint32_t kDfbRemapperPollPauseCycles = 100;

// Per-phase timers inside the init functions (register writes, blob reads, ...). They sit inside the
// e2e window, so they are live only with DFB_INIT_INNER_TIMING_ENABLED on top of
// DFB_INIT_TIMING_ENABLED (TT_METAL_MEASURE_DFB_INIT_INNER_TIME=1). Otherwise rdcycle() is a constant 0
// and every bracket, accumulator and counter guarded by kDfbInitInnerTiming folds away, leaving the
// e2e window exactly the untimed code.
#if defined(DFB_INIT_TIMING_ENABLED) && defined(DFB_INIT_INNER_TIMING_ENABLED)
constexpr bool kDfbInitInnerTiming = true;
#else
constexpr bool kDfbInitInnerTiming = false;
#endif

inline uint32_t rdcycle() {
    if constexpr (kDfbInitInnerTiming) {
        uint32_t c;
        asm volatile("rdcycle %0" : "=r"(c));
        return c;
    }
    return 0;
}

FORCE_INLINE void dfb_remapper_poll_pause() {
    uint32_t t0, t;
    asm volatile("rdcycle %0" : "=r"(t0));
    do {
        asm volatile("rdcycle %0" : "=r"(t));
    } while (t - t0 < kDfbRemapperPollPauseCycles);
}

// Poll until all producers for DFB `dfb_id` have published their signal byte and
// (for DMs with implicit sync) DM0 has armed the ISR.
//
// Signal region layout at dfb_signal_region_off:
//   uint8_t  dfb_signal[NUM_DFBS * MAX_PRODUCERS_PER_DFB]  — per-hart byte slots (192B)
//   uint32_t dfb_expected_signal[NUM_DFBS]                 — host-computed hart mask (128B)
//
// A producer hart h writes byte 1 to dfb_signal[h] once, after its whole init loop (plain
// volatile, no AMO). Consumer reads dfb_expected_signal[d] (mask of the harts producing d), then
// polls each set bit's byte slot.
//
// Runs for DMs unconditionally, and for TRISCs only when compiling for unpack or pack.
// Math (TRISC1) and TRISC3 compile this to an empty function so they never spin on
// uninitialised config data and never block DM0's wait_subordinates().
FORCE_INLINE void dfb_ensure_ready(uintptr_t config_cached, uint8_t dfb_id) {
#if defined(COMPILE_FOR_TRISC) && !defined(UCK_CHLKC_UNPACK) && !defined(UCK_CHLKC_PACK)
    return;
#endif
    const uint32_t sig_off =
        *dfb_l1_uncached_u16_ptr(config_cached + offsetof(dfb_global_header_t, dfb_signal_region_off));

    // dfb_expected_signal[dfb_id] sits after all producer byte slots.
    constexpr uint32_t kSlotStride = static_cast<uint32_t>(::dfb::MAX_PRODUCERS_PER_DFB);
    constexpr uint32_t kExpectedBase = static_cast<uint32_t>(::dfb::NUM_DFBS) * kSlotStride;
    const volatile uint32_t* exp_ptr = reinterpret_cast<const volatile uint32_t*>(dfb_l1_uncached_byte_ptr(
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
    // Poll each producer hart's byte slot; remove from remaining when non-zero.
    uint32_t remaining = expected;
    while (remaining != 0u) {
        const uint8_t bit = static_cast<uint8_t>(__builtin_ctz(remaining));
        const volatile uint8_t* slot = dfb_l1_uncached_byte_ptr(config_cached + sig_off + bit);
        if (*slot != 0u) {
            remaining &= remaining - 1u;
        }
    }
#ifndef COMPILE_FOR_TRISC
    if (need_isr_gate) {
        while (*isr_ready_ptr != 1u) {
        }
    }
#endif
    WAYPOINT("DFD");
}
#ifndef COMPILE_FOR_TRISC

FORCE_INLINE void setup_dfb_implicit_sync(uint32_t tt_l1_ptr* dfb_config_base, uint32_t num_dfbs) {
    // No DFB config was serialized; host will not read timing slots. Still disarm ISR
    // so a prior launch cannot leak into this one.
    if (num_dfbs == 0) {
        disable_dfb_tile_isr();
        return;
    }

    const uint32_t start_time = rdcycle_e2e();

    const uintptr_t config_cached = reinterpret_cast<uintptr_t>(dfb_config_base);
    // The whole prologue is this one uncached load: blob offset, txn-id masks and descriptor
    // stride. 0 means no DM0 ISR on this core.
    const uint64_t dm0_packed = *reinterpret_cast<volatile uint64_t*>(
        dfb_l1_uncached_byte_ptr(config_cached + offsetof(dfb_global_header_t, dm0_packed)));
    if (dm0_packed == 0u) {
        disable_dfb_tile_isr();
        // Header is valid; preserve prior "DM0 finished setup" signaling for consumers
        // that only gate on has_dm0_isr (they will not wait on ready).
        *dfb_l1_uncached_byte_ptr(config_cached + offsetof(dfb_global_header_t, dm0_isr_ready)) = 1u;
        asm volatile("fence w, w" ::: "memory");
        const uint32_t end_time = rdcycle_e2e();
        dfb_init_timing_write_slot(
            0, dfb::DFB_INIT_TIMING_ROLE_DM0_ISR, end_time - start_time, 0, 0, 0, 0, 0, 0, start_time, end_time);
        return;
    }

    const uintptr_t dm0_blob_base = config_cached + dfb_dm0_packed_blob_offset(dm0_packed);
    const uint32_t producer_txn_id_mask = dfb_dm0_packed_producer_mask(dm0_packed);
    const uint32_t consumer_txn_id_mask = dfb_dm0_packed_consumer_mask(dm0_packed);
    const uint32_t all_mask = producer_txn_id_mask | consumer_txn_id_mask;

    WAYPOINT("IS1");

    const uint32_t t_before_desc_copy = rdcycle();
    uint32_t thresh_reg_write_cycles = 0;
    uint32_t dm0_reg_writes = 0;

    // One walk over the used txn ids: copy each descriptor into g_txn_dfb_descriptor, then clear
    // and program its tile-count thresholds. The threshold rides in the descriptor slot, so there
    // is no second walk over a separate threshold table.
    if (all_mask != 0u) {
        const uint8_t* s = static_cast<const uint8_t*>(__builtin_assume_aligned(
            reinterpret_cast<const uint8_t*>(dm0_blob_base + DFB_DM0_CACHED_PAYLOAD_OFFSET), 8));
        // Source slots are packed to 8 / 16 / 32 B (host's choice, carried in dm0_packed).
        const uint32_t src_stride = dfb_dm0_packed_desc_stride(dm0_packed);
        const uint32_t thr_off = dfb_dm0_desc_threshold_offset(src_stride);
        uint32_t pending_desc = all_mask;
        do {
            const uint32_t txn_id = static_cast<uint32_t>(__builtin_ctz(pending_desc));
            uint8_t* d = static_cast<uint8_t*>(__builtin_assume_aligned(&g_txn_dfb_descriptor[txn_id], 32));
            // Word 0 holds the counter count and counters 0-5: the common case is this one copy.
            uint64_t w0;
            __builtin_memcpy(&w0, s, 8);
            __builtin_memcpy(d, &w0, 8);
            if (__builtin_expect((w0 & 0xFFu) > 6u, 0)) {
                // > 6 counters: the slot is 16 B (counters 6-13 in word 1) or 32 B (words 1-2).
                uint64_t w1;
                __builtin_memcpy(&w1, s + 8, 8);
                if (src_stride == 32u) {
                    uint64_t w2;
                    __builtin_memcpy(&w2, s + 16, 8);  // both loads before either store
                    __builtin_memcpy(d + 8, &w1, 8);
                    __builtin_memcpy(d + 16, &w2, 8);
                } else {
                    __builtin_memcpy(d + 8, &w1, 8);
                }
            }
            const uint64_t threshold = s[thr_off];
            const uint32_t t_thr = rdcycle();
            if ((producer_txn_id_mask >> txn_id) & 1u) {
                __builtin_riscv_ttrocc_cmdbuf_clear_tiles_to_process_tr_ack(OVERLAY_RD_CMD_BUF, txn_id);
                asm volatile("nop");
                __builtin_riscv_ttrocc_wr_tiles_to_process_thres_tr_ack(txn_id, threshold);
            }
            if ((consumer_txn_id_mask >> txn_id) & 1u) {
                __builtin_riscv_ttrocc_cmdbuf_clear_tiles_to_process_wr_sent(OVERLAY_WR_CMD_BUF, txn_id);
                asm volatile("nop");
                __builtin_riscv_ttrocc_wr_tiles_to_process_thres_wr_sent(txn_id, threshold);
            }
            thresh_reg_write_cycles += rdcycle() - t_thr;
            if constexpr (kDfbInitInnerTiming) {
                // clear + threshold per role
                dm0_reg_writes +=
                    2u * (((producer_txn_id_mask >> txn_id) & 1u) + ((consumer_txn_id_mask >> txn_id) & 1u));
            }
            // Every stride is a multiple of 8; restate it, or the 8 B loads degrade to 4 B pieces.
            s = static_cast<const uint8_t*>(__builtin_assume_aligned(s + src_stride, 8));
            pending_desc &= (pending_desc - 1u);
        } while (pending_desc);
    }
    const uint32_t t_after_desc_copy = rdcycle();

    const uint32_t t_before_ie = rdcycle();
    uint64_t reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8);
    reg_val = (reg_val & 0x00000000FFFFFFFFULL) | ((uint64_t)(producer_txn_id_mask & 0xFFFFFFFFULL) << 32);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_RD_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_1_REG_OFFSET / 8, reg_val);
    const uint32_t t_after_first_ie_rmw = rdcycle();

    reg_val = __builtin_riscv_ttrocc_cmdbuf_rd_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8);
    reg_val = (reg_val & 0xFFFFFFFF00000000ULL) | (consumer_txn_id_mask & 0xFFFFFFFFULL);
    __builtin_riscv_ttrocc_cmdbuf_wr_reg(
        OVERLAY_WR_CMD_BUF, TT_ROCC_ACCEL_TT_ROCC_CPU0_CMD_BUF_R_PER_TR_ID_IE_2_REG_OFFSET / 8, reg_val);
    const uint32_t t_after_ie_writes = rdcycle();
    if constexpr (kDfbInitInnerTiming) {
        dm0_reg_writes += 4u;  // two IE read-modify-writes
    }

    if (all_mask != 0) {
        enable_dfb_tile_isr();
    } else {
        disable_dfb_tile_isr();
    }
    const uint32_t t_after_isr_enable = rdcycle();

    *dfb_l1_uncached_byte_ptr(config_cached + offsetof(dfb_global_header_t, dm0_isr_ready)) = 1u;
    asm volatile("fence w, w" ::: "memory");

    WAYPOINT("ISD");

    const uint32_t end_time = rdcycle_e2e();
    // Inner metrics are 0 unless kDfbInitInnerTiming.
    const uint32_t hw_reg_write_cycles = thresh_reg_write_cycles + (t_after_ie_writes - t_before_ie);
    dfb_init_timing_write_slot(
        0,
        dfb::DFB_INIT_TIMING_ROLE_DM0_ISR,
        end_time - start_time,
        kDfbInitInnerTiming ? t_before_desc_copy - start_time : 0u,  // A: pre_loop_sw
        t_after_desc_copy - t_before_desc_copy,  // B: subpassB_desc (descriptor copy + threshold writes)
        hw_reg_write_cycles,                     // C: hw_reg_write_cycles
        0,                                       // D: subpassB_l1_read (no separate threshold walk)
        0,                                       // E: subpassB_rocc_issue (no separate threshold walk)
        t_after_first_ie_rmw - t_before_ie,      // F: first_ie_rmw
        start_time,
        end_time,
        t_after_ie_writes - t_after_first_ie_rmw,  // G: second_ie_rmw
        t_after_isr_enable - t_after_ie_writes,    // H: isr_enable
        dm0_reg_writes,                            // I: hw_reg_writes
        thresh_reg_write_cycles);                  // J: subpassB_hw (threshold register writes)
}

FORCE_INLINE void setup_dfb_remapper(uint32_t tt_l1_ptr* dfb_config_base, uint32_t num_dfbs) {
    if (num_dfbs == 0) {
        return;
    }

    const uint32_t start_time = rdcycle_e2e();

    const uintptr_t config_cached = reinterpret_cast<uintptr_t>(dfb_config_base);

    // One uncached load: [15:0] slot array offset from the region base, [31:16] num_slots. The
    // slot array itself is walked cached.
    const uint32_t dm1_slots_word = *dfb_l1_uncached_u32_ptr(config_cached + offsetof(dfb_global_header_t, dm1_slots));
    const dfb_dm1_remapper_slot_t* slots =
        reinterpret_cast<const dfb_dm1_remapper_slot_t*>(config_cached + (dm1_slots_word & 0xFFFFu));
    const uint16_t num_slots = static_cast<uint16_t>(dm1_slots_word >> 16);

    WAYPOINT("RS");

    // Inner timers (0 unless kDfbInitInnerTiming).
    uint32_t blob_l1_read_sw = 0;
    uint32_t pairs_reg_hw = 0;
    uint32_t first_pair_clientR_hw = 0;
    uint32_t first_pair_clientL_hw = 0;
    uint32_t last_pair_hw = 0;
    const uint32_t t_loop_start = rdcycle();

    uint32_t max_pair_idx = 0;
    for (uint16_t s = 0; s < num_slots; s++) {
        const uint32_t t_read = rdcycle();
        const uint32_t pair_idx = slots[s].pair_index;
        const uint32_t clientR_val = slots[s].clientR_val;
        const uint32_t clientL_val = slots[s].clientL_val;
        const uint32_t t_clientR = rdcycle();
        WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pair_idx), clientR_val);
        const uint32_t t_clientL = rdcycle();
        WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx), clientL_val);
        const uint32_t t_done = rdcycle();
        if constexpr (kDfbInitInnerTiming) {
            blob_l1_read_sw += t_clientR - t_read;
            last_pair_hw = t_done - t_clientR;
            pairs_reg_hw += last_pair_hw;
            if (s == 0) {
                first_pair_clientR_hw = t_clientL - t_clientR;
                first_pair_clientL_hw = t_done - t_clientL;
            }
        }
        if (pair_idx > max_pair_idx) {
            max_pair_idx = pair_idx;
        }
    }
    const uint32_t t_loop_end = rdcycle();

    if (num_slots > 0u) {
        g_remapper_configurator.set_pair_high_watermark(max_pair_idx);
    }

    WAYPOINT("RSD");
    const uint32_t end_time = rdcycle_e2e();

    dfb_init_timing_write_slot(
        1,
        dfb::DFB_INIT_TIMING_ROLE_DM1_RMP,
        end_time - start_time,
        blob_l1_read_sw,                                               // A: blob_l1_read_sw
        (t_loop_end - t_loop_start) - blob_l1_read_sw - pairs_reg_hw,  // B: blob_loop_ovhd
        pairs_reg_hw,                                                  // C: pairs_reg_hw
        0,                                                             // D: enable_remapper_hw (none)
        first_pair_clientR_hw,                                         // E
        first_pair_clientL_hw,                                         // F
        start_time,
        end_time,
        last_pair_hw,    // G: last_pair_hw
        pairs_reg_hw,    // H: hw_reg_write_cycles
        num_slots * 2u,  // I: hw_reg_writes (clientR + clientL per slot)
        num_slots);      // J: pairs_slots_written
}

#endif  // !COMPILE_FOR_TRISC

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)

// Intra-tensix alias (HW workaround): route this Neo's ClientL tile counter to a sacrificial
// ClientR shadow counter, so the T6 counter update copy lands there instead of aliasing into
// overlay counters 0-15. Packer and unpacker both keep driving the ClientL counter; nothing reads
// the shadow. ClientL and ClientR need not be adjacent — remapper maps any (id, cnt_sel) pair.
//
// This cannot be done by DM1 like every other remapper pair: the Tensix-only counter pool is
// invisible to DM, and one Neo cannot see another Neo's counters. So each Neo's packer programs
// its own pair.
FORCE_INLINE void dfb_program_intra_tensix_alias(
    uint32_t pair_idx, uint32_t client_tc_id, uint32_t shadow_tc_id, uint32_t neo_id) {
    const uint32_t client_id = static_cast<uint32_t>(overlay::NEO_0) + neo_id;
    const uint32_t clientR_val = (client_id & 0x7u) | ((shadow_tc_id & 0x1Fu) << 3);
    const uint32_t clientL_val = (client_id & 0x7u) | ((client_tc_id & 0x1Fu) << 3) |
                                 (0x1u << 8) |  // valid: ClientR slot 0 only
                                 (0x1u << 12);  // clientl_is_producer=1; clientr_group=0, distribute=0

    // Publish ClientR first, then ClientL (which carries the valid bits)
    WRITE_REG32(REMAP_CLIENT_R_CONFIG_REG_ADDR32(pair_idx), clientR_val);
    WRITE_REG32(REMAP_CLIENT_L_CONFIG_REG_ADDR32(pair_idx), clientL_val);
}

#endif  // COMPILE_FOR_TRISC && UCK_CHLKC_PACK

// DM2-7 + TRISC: walk this hart's pre-computed sequential init blob; TC readiness is published
// with one store per producer hart and consumers poll in DataflowBuffer::DataflowBuffer().
FORCE_INLINE DfbPackerRemapperRange setup_local_dfb_interfaces(uint32_t tt_l1_ptr* dfb_config_base, uint32_t num_dfbs) {
    DfbPackerRemapperRange packer_rmp{};
    if (num_dfbs == 0) {
        return packer_rmp;
    }

    const uint32_t start_time = rdcycle_e2e();

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

    // ONE uncached load for the whole prologue: entry count, producer flag, blob start and the
    // signal-region offset (dfb_hart_desc_t).
    const uint64_t hart_desc = dfb_read_hart_desc(config_cached, hart_u8);
    const uint32_t num_init = static_cast<uint32_t>(hart_desc) & 0xFFu;
    const uint32_t hart_blob_start = static_cast<uint32_t>(hart_desc >> 16) & 0xFFFFu;
    const uint32_t dfb_signal_region_off = static_cast<uint32_t>(hart_desc >> 48);

    volatile uint8_t* const dfb_signal_base = dfb_l1_uncached_byte_ptr(config_cached + dfb_signal_region_off);

    const volatile uint8_t* p = reinterpret_cast<const volatile uint8_t*>(config_cached + hart_blob_start);

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    uint8_t compact_dfb_count = 0;
    uint8_t pack_tc_slot_count = 0;
#endif

    // Inner timers and counters (0 unless kDfbInitInnerTiming).
    uint32_t total_entry_hdr = 0;
    uint32_t total_tc_slots = 0;
    uint32_t total_remapper_spin = 0;
    uint32_t total_tc_hw = 0;
    uint32_t total_tc_reset_hw = 0;
    uint32_t total_tc_capacity_hw = 0;
    uint32_t total_hw_reg_writes = 0;
    const uint32_t t_loop_start = rdcycle();

    // -----------------------------------------------------------------------
    // Sequential init blob walk: no pointer-table lookups, no per-DFB indirection.
    // DM walks with cached TL1 pointers (the host writes the config snoopable, so no L2
    // invalidate is needed); TRISC reads through the uncached alias.
    // -----------------------------------------------------------------------

    // Widen walk counters to uint32_t so RV64 codegen avoids zext.b/zext.w on every
    // increment / compare. Blob fields stay uint8_t; only locals are widened.
    for (uint32_t i = 0; i < num_init; i++) {
        const uintptr_t e_addr = reinterpret_cast<uintptr_t>(p);
        const uint32_t t_hdr_start = rdcycle();

#ifndef COMPILE_FOR_TRISC
        // Read the 8B control prefix only. The interface image that follows is never decoded --
        // it is copied verbatim below.
        const uint32_t c0 = *reinterpret_cast<const uint32_t*>(e_addr);
        const uint32_t c1 = *reinterpret_cast<const uint32_t*>(e_addr + 4u);
        const uint32_t iface_byte_off = c0 & 0xFFFFu;
        const uint32_t flags = (c0 >> 16) & 0xFFu;
        const uint32_t capacity = c1 & 0xFFFFu;
        const uint32_t remapper_pair_index = (c1 >> 16) & 0xFFu;
        // Byte 7 carries the finished entry stride, so advancing to the next entry is an add rather
        // than a multiply-add on the pointer's dependency chain.
        const uint32_t entry_bytes = c1 >> 24;
        // num_tcs from the image's own num_tcs_to_rr byte, which the copy below loads anyway.
        const uint32_t num_tcs =
            *reinterpret_cast<const uint8_t*>(e_addr + DFB_IFACE_IMAGE_CTRL_BYTES + DFB_DM_IFACE_NUM_TCS_BYTE_OFF);
#else
        const dfb_init_entry_hdr_t eh = dfb_read_init_entry_header(e_addr);
        const uint32_t flags = eh.flags;
        const uint32_t capacity = eh.capacity;
        const uint32_t remapper_pair_index = eh.remapper_pair_index;
        const uint32_t num_tcs = eh.num_tcs;
        // Advance to the next entry: 32B header + ((num_tcs*9 + 3) & ~3).
        const uint32_t entry_bytes = dfb_hart_init_entry_byte_size(num_tcs);
        // AoP TC tail starts right after the 32B header: pairs first, ptc bytes after all pairs.
        const uintptr_t tc_base_addr = e_addr + sizeof(dfb_hart_init_entry_t);
#endif
        p = reinterpret_cast<const volatile uint8_t*>(e_addr + entry_bytes);
        total_entry_hdr += rdcycle() - t_hdr_start;

        WAYPOINT("L1");

        // --- Resolve g_dfb_interface slot ---
#ifndef COMPILE_FOR_TRISC
        // Host already multiplied by sizeof(LocalDFBInterface): base + offset, no multiply here.
        LocalDFBInterface& iface =
            *reinterpret_cast<LocalDFBInterface*>(reinterpret_cast<uintptr_t>(g_dfb_interface) + iface_byte_off);
#elif defined(UCK_CHLKC_PACK)
        ASSERT(compact_dfb_count < dfb::MAX_ACTIVE_DFBS_PACK);
        ASSERT(num_tcs <= dfb::MAX_NUM_TILE_COUNTERS_TO_RR);
        ASSERT(pack_tc_slot_count + num_tcs <= dfb::MAX_PACK_TC_SLOTS);
        const uint8_t compact_id = compact_dfb_count++;
        g_dfb_logical_to_compact[eh.logical_dfb_id] = compact_id;
        LocalDFBInterface& iface = g_dfb_interface[compact_id];
        iface.tc_slots.slot_base = pack_tc_slot_count;
        pack_tc_slot_count += static_cast<uint8_t>(num_tcs);
#else
        LocalDFBInterface& iface = g_dfb_interface[eh.logical_dfb_id];
#endif

        WAYPOINT("L2");

#ifdef COMPILE_FOR_TRISC
        // --- Scalar fields ---
        iface.num_tcs_to_rr = static_cast<uint8_t>(num_tcs);
        iface.tc_idx = 0;
        const uint32_t entry_size_units = static_cast<uint16_t>(eh.entry_size >> cb_addr_shift);
        iface.entry_size = static_cast<uint16_t>(entry_size_units);
        // Host precomputes stride in tile units: (entry_size >> shift) * stride_in_entries.
        iface.stride_size = static_cast<uint16_t>(eh.stride_size_precomp);
        iface.stride_size_tiles = eh.stride_size_tiles;
        iface.block_size = eh.block_size;
        iface.split_tc = eh.split_tc;
        iface.jump = eh.entries_to_jump;
        ASSERT(iface.stride_size_tiles != 0 && iface.block_size != 0);
#if defined(UCK_CHLKC_PACK)
        iface.wr_entry_ptr = 0;
#else  // unpack TRISC
        iface.tensix_trisc_mask = static_cast<uint8_t>(eh.flags & DFB_HART_FLAG_TRISC_MASK);
#endif
        iface.num_entries = eh.num_entries;

        WAYPOINT("L3");

        // --- TC slot population ---
        // AoP TC tail: dfb_blob_tc_pair_t[num_tcs] (8B each, base+limit adjacent per slot)
        // immediately after the 32B header, then uint8_t ptc[num_tcs] padded to 4B.
        // Preload ptc as 1–2 word reads before the loop; in-loop ptc is a register bit-extract.
        const uintptr_t ptc_base_addr = tc_base_addr + (static_cast<uintptr_t>(num_tcs) << 3u);
        const volatile dfb_blob_tc_pair_t* pairs =
            reinterpret_cast<const volatile dfb_blob_tc_pair_t*>(dfb_l1_uncached_u32_ptr(tc_base_addr));
        const uint32_t ptc_w0 = (num_tcs > 0u) ? *dfb_l1_uncached_u32_ptr(ptc_base_addr) : 0u;
        const uint32_t ptc_w1 = (num_tcs > 4u) ? *dfb_l1_uncached_u32_ptr(ptc_base_addr + 4u) : 0u;
        // Slot 0's base and the entry size stay in registers: the slot stores below may alias the
        // slot they would be read back from, so the compiler would reload them for every slot.
        // Slot 0's index is 0 by definition, so its divide is skipped outright.
        uint32_t base0 = 0;
        const uint32_t t_slots_start = rdcycle();
        for (uint32_t t = 0; t < num_tcs; t++, pairs++) {
            // Host pre-shifts TRISC TC addresses to tile units at serialization.
            const uint32_t base = pairs->base_addr;
            const uint32_t limit = pairs->limit;
            const uint8_t ptc = static_cast<uint8_t>((t < 4u ? ptc_w0 : ptc_w1) >> ((t & 3u) * 8u));
            uint16_t base_entry_idx = 0;
            if (t == 0u) {
                base0 = base;
            } else {
                base_entry_idx = static_cast<uint16_t>((base - base0) / entry_size_units);
            }
            iface.tc_slots[t].packed_tile_counter = ptc;
            iface.tc_slots[t].base_addr = base;
            // ring_size is uint32 L1 units (full Quasar L1).
            iface.tc_slots[t].ring_size = limit - base;
            iface.tc_slots[t].base_entry_idx = base_entry_idx;
#if defined(UCK_CHLKC_PACK)
            // Cursor byte-offset is derived from wr_entry_idx (not stored).
            iface.tc_slots[t].wr_entry_idx = base_entry_idx;
#else  // unpack
            iface.tc_slots[t].rd_entry_idx = base_entry_idx;
#endif
        }
        total_tc_slots += rdcycle() - t_slots_start;
#else
        // The entire interface -- scalars AND every tc_slot -- arrives as one contiguous host-built
        // image (dfb_write_dm_image_entry). Pure load/store.
        {
            static_assert(
                offsetof(LocalDFBInterface, tc_slots) == DFB_DM_IFACE_SCALAR_BYTES,
                "DM iface scalar head must match DFB_DM_IFACE_SCALAR_BYTES");
            static_assert(sizeof(DFBTCSlot) == DFB_DM_IFACE_SLOT_BYTES, "DFBTCSlot must match DFB_DM_IFACE_SLOT_BYTES");
            static_assert(
                sizeof(LocalDFBInterface) == DFB_DM_IFACE_SIZE, "LocalDFBInterface must match DFB_DM_IFACE_SIZE");
            static_assert(
                offsetof(LocalDFBInterface, num_tcs_to_rr) == DFB_DM_IFACE_NUM_TCS_BYTE_OFF,
                "num_tcs_to_rr must match DFB_DM_IFACE_NUM_TCS_BYTE_OFF");

            // 64-bit copy: both sides 8B-aligned (every entry is a multiple of 8; iface alignas(64)).
            const uint64_t* src64 = static_cast<const uint64_t*>(
                __builtin_assume_aligned(reinterpret_cast<const void*>(e_addr + DFB_IFACE_IMAGE_CTRL_BYTES), 8));
            uint64_t* dst64 = static_cast<uint64_t*>(__builtin_assume_aligned(&iface, 8));
            const uint32_t t_slots_start = rdcycle();

            // Fixed-size 32B scalar head: straight-line, no trip count, no branch.
            dst64[0] = src64[0];
            dst64[1] = src64[1];
            dst64[2] = src64[2];
            dst64[3] = src64[3];

            // One trip per TC slot with the slot unrolled inside, not one trip per word: num_tcs is
            // a runtime value, and a word-at-a-time loop pays increment/compare/branch per word.
            for (uint32_t t = 0; t < num_tcs; t++) {
                dst64[4 + 2 * t] = src64[4 + 2 * t];  // ptr, base_addr
                dst64[5 + 2 * t] = src64[5 + 2 * t];  // limit, packed_tile_counter
            }
            total_tc_slots += rdcycle() - t_slots_start;
        }
        ASSERT(iface.entry_size != 0 && iface.stride_size >= iface.entry_size && iface.block_size != 0);
#endif

        WAYPOINT("L4");

        // --- Producer-only: wait for remapper pair + TC HW init ---
#if !defined(COMPILE_FOR_TRISC) || defined(UCK_CHLKC_PACK)
        if (flags & DFB_HART_FLAG_IS_PRODUCER) {
            // A producer's remapper pair is programmed one of two ways; host sets at most one flag.
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
            // Intra-tensix: this packer owns the pair, so it programs rather than waits. Must happen
            // before the TC reset/capacity writes below so the alias is live for the first update.
            if (flags & DFB_HART_FLAG_REMAPPER_SELF_PROG) {
                dfb_program_intra_tensix_alias(
                    remapper_pair_index,
                    dfb::get_counter_id(iface.tc_slots[0].packed_tile_counter),
                    eh.intra_shadow_tc_id,
                    neo_id);
                // Contiguous [lo, hi) over this Neo's packer-owned pairs (host assigns ascending
                // indices within the reserved block). Teardown clears exactly this window.
                if (packer_rmp.lo == 0xFFu) {
                    packer_rmp.lo = static_cast<uint8_t>(remapper_pair_index);
                }
                packer_rmp.hi = static_cast<uint8_t>(remapper_pair_index + 1u);
            }
#endif
            // Remapped producers: spin until DM1 has written this pair's ClientL config with
            // non-zero valid bits (bits [11:8]). No global remapper-enable wait is needed.
            if (flags & DFB_HART_FLAG_REMAPPER_WAIT_DM1) {
                const uint32_t t_spin = rdcycle();
                WAYPOINT("RMSW");
                while (overlay::RemapperAPI::get_clientL_valid_hw(remapper_pair_index) == 0u) {
                    dfb_remapper_poll_pause();
                }
                WAYPOINT("RMSD");
                total_remapper_spin += rdcycle() - t_spin;
            }

            const uint32_t t_tc_hw = rdcycle();

            // One pass per tile counter: reset, then set capacity. Each counter's reset is fenced
            // before its own capacity write; the counters are independent of each other.
            for (uint32_t t = 0; t < num_tcs; t++) {
                const uint8_t packed_ptc = iface.tc_slots[t].packed_tile_counter;
                const uint8_t tc_id = dfb::get_counter_id(packed_ptc);
                const uint32_t t_reset = rdcycle();
#ifndef COMPILE_FOR_TRISC
                const uint8_t tensix_id = dfb::get_tensix_id(packed_ptc);
                overlay::fast_llk_intf_reset(tensix_id, tc_id);
                asm volatile("fence w, w" ::: "memory");
                const uint32_t t_cap = rdcycle();
                overlay::fast_llk_intf_set_capacity(tensix_id, tc_id, capacity);
#else
                ckernel::trisc::tile_counters[tc_id].f.reset = 1;
                const uint32_t t_cap = rdcycle();
                ckernel::trisc::tile_counters[tc_id].f.buf_capacity = capacity;
#endif
                const uint32_t t_cap_done = rdcycle();
                if constexpr (kDfbInitInnerTiming) {
                    total_tc_reset_hw += t_cap - t_reset;
                    total_tc_capacity_hw += t_cap_done - t_cap;
                    total_hw_reg_writes += 2u;  // reset + capacity
                }
            }
#ifndef COMPILE_FOR_TRISC
            asm volatile("fence w, w" ::: "memory");
#endif
            total_tc_hw += rdcycle() - t_tc_hw;
        }
#endif  // !COMPILE_FOR_TRISC || UCK_CHLKC_PACK

        WAYPOINT("L5");
    }

    const uint32_t t_after_loop = rdcycle();

#if !defined(COMPILE_FOR_TRISC) || defined(UCK_CHLKC_PACK)
    // One producer-ready store for the whole hart, after every entry's TC init (each DM entry
    // ends with fence w,w; TRISC keeps compiler-barrier-only ordering). The compiler barrier keeps
    // the store from being hoisted above the TC writes.
    if ((static_cast<uint32_t>(hart_desc >> 8) & DFB_HART_DESC_FLAG_PRODUCER) != 0u) {
        asm volatile("" ::: "memory");
        dfb_signal_base[hart_u8] = 1u;
        WAYPOINT("PPR");
    }
#endif
    const uint32_t t_after_publish = rdcycle();

    WAYPOINT("L12");
    const uint32_t end_time = rdcycle_e2e();

#ifdef COMPILE_FOR_TRISC
    const uint8_t timing_slot = dfb_init_timing_trisc_slot_index();
    const uint8_t timing_role = dfb::DFB_INIT_TIMING_ROLE_TRISC_LOCAL;
#else
    const uint8_t timing_slot = hart_u8;
    const uint8_t timing_role = dfb::DFB_INIT_TIMING_ROLE_DM_LOCAL;
#endif
    // Inner metrics are 0 unless kDfbInitInnerTiming.
    dfb_init_timing_write_slot(
        timing_slot,
        timing_role,
        end_time - start_time,
        kDfbInitInnerTiming ? t_after_loop - start_time : 0u,  // A: merged_sw (prologue + entry loop)
        total_remapper_spin,                                   // B: remapper_spin
        total_tc_hw,                                           // C: tc_hw
        total_hw_reg_writes,                                   // D: hw_reg_writes (TC reset + capacity per producer TC)
        total_tc_reset_hw,                                     // E: tc_reset_hw
        total_tc_capacity_hw,                                  // F: tc_capacity_hw
        start_time,
        end_time,
        kDfbInitInnerTiming ? t_loop_start - start_time : 0u,  // G: pre_loop (hart_desc load)
        total_entry_hdr,                                       // H: entry_hdr
        total_tc_slots,                                        // I: tc_slots
        t_after_publish - t_after_loop);                       // J: sig_write (per-hart producer-ready store)
    return packer_rmp;
}
