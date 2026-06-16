// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace dfb {

enum AccessPattern : uint8_t {
    STRIDED,
    ALL,
    UNKNOWN,
};

constexpr uint8_t NUM_DFBS = 32;
// Pack TRISC stores only active logical DFBs in a compact local array to reduce local-memory pressure.
constexpr uint8_t MAX_ACTIVE_DFBS_PACK = 16;

constexpr uint8_t NUM_TENSIX = 4;
constexpr uint8_t NUM_TILE_COUNTERS_PER_TENSIX = 32;
constexpr uint8_t NUM_TENSIX_TILE_COUNTERS_FOR_DM = 16;
// First TC ID in the default Tensix-only pool (not accessible by DM); used for intra/inter-tensix DFBs.
// Note: The Remapper can be programmed to expose these TCs to DMs.
constexpr uint8_t TC_TENSIX_POOL_START = NUM_TENSIX_TILE_COUNTERS_FOR_DM;  // = 16
constexpr uint8_t NUM_REMAPPER_PAIRINGS = 64;
constexpr uint8_t NUM_TXN_IDS = 4;
constexpr uint8_t MAX_NUM_TILE_COUNTERS_TO_RR = 6;
// DM0 blob constants
constexpr uint8_t MAX_DM0_REMAPPER_SLOTS = 8;  // max DM producer RISCs
constexpr uint8_t MAX_CLIENT_RS          = 4;   // max consumers per remapper slot (4 Tensix or 4 DM clientR IDs)
constexpr uint8_t MAX_TCS_PER_TXN        = 8;   // max producer TCs per txn entry (8 DM producers × 1 TC each)

constexpr uint16_t TENSIX_RISC_OFFSET = 8; // First 8 represent DMs
// Hartids 0-7 = DM0-7, 8-11 = Neo0-3 (TRISC init uses 8 + neo_id).
constexpr uint8_t NUM_PARTICIPATING_HARTIDS = 12;

using PackedTileCounter = uint8_t;  // bits 5-6: tensix_id (2 bits), bits 0-4: counter_id (5 bits)

// PackedTileCounter bit layout constants
constexpr uint8_t PACKED_TC_COUNTER_ID_BITS = 5;  // Number of bits for counter_id
constexpr uint8_t PACKED_TC_COUNTER_ID_MASK =
    (1 << PACKED_TC_COUNTER_ID_BITS) - 1;                                 // 0x1F - mask for 5-bit counter_id (0-31)
constexpr uint8_t PACKED_TC_TENSIX_ID_SHIFT = PACKED_TC_COUNTER_ID_BITS;  // 5 - shift to access tensix_id
constexpr uint8_t PACKED_TC_TENSIX_ID_BITS = 2;                           // Number of bits for tensix_id
constexpr uint8_t PACKED_TC_TENSIX_ID_MASK =
    (1 << PACKED_TC_TENSIX_ID_BITS) - 1;  // 0x03 - mask for 2-bit tensix_id (0-3)

// NOLINTBEGIN(readability-redundant-inline-specifier)
inline __attribute__((always_inline)) constexpr uint8_t get_tensix_id(PackedTileCounter p) {
    return (p >> PACKED_TC_TENSIX_ID_SHIFT) & PACKED_TC_TENSIX_ID_MASK;
}

inline __attribute__((always_inline)) constexpr uint8_t get_counter_id(PackedTileCounter p) {
    return p & PACKED_TC_COUNTER_ID_MASK;
}
// NOLINTEND(readability-redundant-inline-specifier)

}  // namespace dfb

/*
    DFB config region layout (Quasar / tt-2xx):

    [dfb_config_base]:
      dfb_global_header_t (96B) — fixed-size; DM1/DM0 blob offsets stored inside.

    [dfb_config_base + dm1_remapper_blob_offset]:
      DM1 remapper blob — contiguous across all DFBs, read by DM1:
        [DFB0: dfb_dm1_remapper_entry_header_t(4B) + dfb_dm0_remapper_slot_t × n]
        [DFB1: ...]
        ...

    [dfb_config_base + dm0_isr_blob_offset]:
      DM0 ISR blob — core-wide, read by DM0:
        [dfb_dm0_isr_blob_core_header_t(8B): precomputed producer/consumer txn IE masks]
        [txn_threshold_pool: dfb_dm0_isr_txn_threshold_t indexed by txn_id, span 0..max_txn_id]
        [txn_desc_pool: dfb_dm0_txn_descriptor_image_t indexed by txn_id, contiguous after txn_hw_pool]
        ...

    [dfb_config_base + ghdr->hart_blob_offset[h]]:
      Per-hart sequential init blob (one per participating hartid):
        dfb_hart_init_entry_t[num_entries]  — one per DFB this hart participates in
        (4B-padded end)
      Entry count is NOT stored in the blob; device derives it from participation_mask[h].
      hart_blob_offset[h] points at the first init entry (4B-aligned).
      Non-participating harts have a minimal 4-byte {0,0,0,0} blob.

    [dfb_config_base + dfb_signal_region_off]:
      uint8_t  dfb_signal[NUM_DFBS * MAX_NUM_TILE_COUNTERS_TO_RR]  — 192B; producer i of DFB d writes
                                                                       byte 1 to slot [d*MAX_PRODUCERS+i].
      uint32_t dfb_expected_signal[NUM_DFBS]                        — 128B; host-computed bitmask of which
                                                                       producer bits are active per DFB.
      Producers use a plain volatile store + fence (no AMO). Consumers iterate over bits in
      dfb_expected_signal[d] and poll each producer's byte slot in dfb_signal.

    DM1 reads linearly through only remapper slot data (unchanged).
    DM0 reads linearly through only ISR txn data (unchanged).
    DM2-7 + TRISC each walk their own sequential init blob — no pointer-table indirection.

    Memory (worst case 4Sx4A, 5 riscs, 4 rmp slots, 8 DFBs):
      96 + (4+4*16)*8 + (4+20)*8 + per_hart_blobs(~3.2KB) + signal_region(256B) ≈ 3.8KB
*/

// Fixed header at the start of the DFB config region.
struct dfb_global_header_t {
    uint32_t dm1_remapper_blob_offset;  // → DM1 remapper blob
    uint32_t dm0_isr_blob_offset;       // → DM0 ISR blob (core header + txn pools)
    uint32_t dfb_signal_region_off;     // → signal region: per-producer byte slots then dfb_expected_signal[NUM_DFBS]
    uint8_t  num_dfbs;
    uint8_t  dm0_isr_ready;             // cleared by host; set by DM0 when ISR is armed
    uint8_t  has_dm0_isr;               // 1 if any DFB uses implicit sync (replaces per_dfb_layout_offset > dm0 check)
    uint8_t  _pad;
    // participation_mask[h] bit i set → hartid h participates in DFB i.
    // Device init uses popcount(participation_mask[h]) as this hart's init/wait entry count.
    uint32_t participation_mask[dfb::NUM_PARTICIPATING_HARTIDS];  // 48B
    // Byte offset from config_base to each hartid's init+wait blob (first init entry).
    // 0 for non-participating harts (minimal {0,0,0,0} blob is still emitted).
    uint16_t hart_blob_offset[dfb::NUM_PARTICIPATING_HARTIDS];    // 24B
    uint8_t  _pad2[8];  // pad to 96B
};

// DM1/DM0 blobs begin immediately after the header; no prefix tables.
inline uint32_t dfb_config_header_size() { return sizeof(dfb_global_header_t); }

// Number of init/wait entries in a hart blob (= popcount of participation_mask[h]).
inline uint8_t dfb_hart_participation_count(uint32_t participation_mask) {
    return static_cast<uint8_t>(__builtin_popcount(participation_mask));
}

// DM1 tile-counter init option (device firmware only):
//   0 = baseline (default): each remapped producer spins on remapper-enable, then resets its own
//       tile counter, sets capacity, and publishes its readiness from setup_local_dfb_interfaces.
//   2 = DM1 enables the remapper first, then resets each remapped producer's tile counter, sets
//       capacity, and publishes readiness on the producer's behalf (setup_dfb_remapper). The
//       producer skips its own remapper spin / TC init / publish and waits on the published
//       signal in DataflowBuffer's ctor (dfb_ensure_ready).
#ifndef DFB_DM1_TC_INIT_OPTION
#define DFB_DM1_TC_INIT_OPTION 2
#endif

// Flag bits for dfb_hart_init_entry_t::flags
constexpr uint8_t DFB_HART_FLAG_IS_PRODUCER  = (1u << 7);
constexpr uint8_t DFB_HART_FLAG_REMAPPER_EN  = (1u << 6);
constexpr uint8_t DFB_HART_FLAG_BROADCAST_TC = (1u << 5);
constexpr uint8_t DFB_HART_FLAG_TRISC_MASK   = 0x0Fu;  // bits 3:0 = tensix_trisc_mask (which TRISC(s) run DFB ops)

// Per-(hart, DFB) init entry in this hart's sequential blob.
// Fixed 24B header immediately followed (4B-aligned) by variable TC address arrays:
//   uint32_t tc_base_bytes[num_tcs]  — raw byte addresses; device applies >> cb_addr_shift
//   uint32_t tc_limit_bytes[num_tcs] — raw byte addresses; device applies >> cb_addr_shift
//   uint8_t  packed_tc[num_tcs]
//   uint8_t  _pad[] → next 4B boundary
struct dfb_hart_init_entry_t {
    uint8_t  logical_dfb_id;
    uint8_t  num_tcs;
    uint8_t  flags;                          // DFB_HART_FLAG_* bits above; bits3:0 = tensix_trisc_mask
    uint8_t  capacity;                       // producer: TC capacity; consumer: 0
    uint32_t entry_size;                     // raw bytes; device applies >> cb_addr_shift
    uint32_t stride_in_entries;              // device: stride_size = (entry_size >> shift) * stride_in_entries
    uint8_t  stride_size_tiles;              // = (uint8_t)stride_in_entries — stored for TRISC direct use
    uint8_t  num_txn_ids;                    // DM only; 0 for TRISC
    uint8_t  threshold;                      // DM only
    uint8_t  num_entries_per_txn_id;         // DM only
    uint8_t  num_entries_per_txn_id_per_tc;  // DM only
    uint8_t  producer_signal_bit;            // index into this DFB's producer slot row (0-based); 0xFF if consumer
    uint8_t  txn_ids[dfb::NUM_TXN_IDS];     // DM only; unused slots zero
    uint8_t  _pad[2];                        // pad header to 24B → 4B-aligned TC arrays follow
} __attribute__((packed));
static_assert(sizeof(dfb_hart_init_entry_t) == 24, "dfb_hart_init_entry_t must be 24B");

// Returns total serialized bytes for one dfb_hart_init_entry_t with num_tcs TC slots.
inline uint32_t dfb_hart_init_entry_byte_size(uint8_t num_tcs) {
    const uint32_t tail =
        static_cast<uint32_t>(num_tcs) * (sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint8_t));
    return static_cast<uint32_t>(sizeof(dfb_hart_init_entry_t)) + ((tail + 3u) & ~3u);
}

struct dfb_txn_id_descriptor_t {
    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t num_entries_to_process_threshold; // entries each txn ID tracks before posting/acking
    uint8_t num_txn_ids;
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
} __attribute__((packed));

// No __attribute__((packed)): every field is naturally aligned at its current offset —
// entry_size/stride_in_entries (u32) at 0/4, capacity (u16) at 8, risc_mask_bits (u16 bitfield) at 10,
// producer/consumer txn descriptors (packed, alignment 1) at 12/20, trailing u8 fields at 28-31.
// Without packed the compiler emits lw for entry_size and stride_in_entries (1 instr vs. 10 lbu+shift+or),
// and lhu for capacity (1 instr vs. 2 lbu). Layout and sizeof are identical — static_assert guards.
struct dfb_initializer_t {
    uint32_t entry_size;
    uint32_t stride_in_entries;
    uint16_t capacity;
    struct {
        uint16_t dm_mask : 8;             // bits 0-7: DM RISC mask
        uint16_t tensix_mask : 4;         // bits 8-11: Neo RISC mask
        uint16_t tensix_trisc_mask : 4;   // bits 12-15: which TRISC(s) on the Neo run DFB ops (see dataflow_buffer.inl)
    } risc_mask_bits;
    // Participant mask (DM/Neo hartids, per_risc layout, popcount): dm_mask | (tensix_mask << 8).
    // tensix_trisc_mask is separate — TRISC-side only, not OR'd into that mask.
    // For DM-to-DM DFBs, producer and consumer would have different set of transaction ids
    dfb_txn_id_descriptor_t producer_txn_descriptor;
    dfb_txn_id_descriptor_t consumer_txn_descriptor;
    uint8_t num_producers;
    uint8_t _pad[3];                  // reserved (was dm0_blob_size; DM0 blob is now a separate global region)
};
static_assert(sizeof(dfb_initializer_t) == 32, "dfb_initializer_t size changed — check field alignment");

// AoS layout: base_addr and limit for the same TC slot are adjacent in memory,
// improving spatial locality in the TC-init loop that accesses both per iteration.
// No __attribute__((packed)): both fields are naturally-aligned uint32_t; packed would force
// byte-by-byte lbu loads (10 instructions per field) instead of single lw instructions.
struct TCAddressEntry {
    uint32_t base_addr;
    uint32_t limit;
};

// No __attribute__((packed)) on the outer struct: all fields are naturally aligned
// (tc_addrs at offset 0 is 4-byte aligned; packed_tile_counter/bitfield structs are 1-byte
// types at byte offsets). Removing packed allows lw for tc_addrs[i].base_addr and .limit
// (2 lw vs. 20 byte-load instructions per slot). Layout is identical — static_assert guards.
struct dfb_initializer_per_risc_t {  // 64 bytes (power-of-2; array index = risc_index << 6)
    TCAddressEntry tc_addrs[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];  // 48 bytes
    dfb::PackedTileCounter packed_tile_counter[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];  // 6 bytes
    struct {
        uint8_t num_tcs_to_rr : 4;   // 0..15, number of TCs to round-robin (max 6 for 6 user-programmable DM cores)
        uint8_t tc_init_done : 1;
        uint8_t broadcast_tc : 1;    // DM-DM ALL: producer posts to all TCs instead of round-robin
        uint8_t reserved : 2;
    } __attribute__((packed)) num_tcs_and_init;
    struct {
        uint8_t reserved : 6;    // bits 0-5: formerly remapper_pair_index (now in DM0 blob)
        uint8_t remapper_en : 1; // bit 6
        uint8_t is_producer : 1; // bit 7: indicates if this RISC is a producer
    } __attribute__((packed)) flags;
    uint8_t _pad[8];  // pad to 64 bytes; per_risc_base[i] = base + (i << 6)
};

// intra tensix dfb
// (24 * 16) * 4 = 1,536 bytes
struct dfb_initializer_intra_tensix_t {  // 24 bytes
    uint32_t logical_id;
    uint32_t entry_size;
    uint32_t stride_size;
    uint32_t base_addr;
    uint32_t limit;
    uint16_t capacity;
    dfb::PackedTileCounter packed_tile_counter;
    uint8_t tensix_mask;
} __attribute__((packed));

// Per-DFB header for the DM1 remapper blob.
// DM1 reads linearly through this region, processing only remapper slot data.
struct dfb_dm1_remapper_entry_header_t {  // 4 bytes
    uint8_t num_remapper_slots;
    uint8_t _pad[3];
} __attribute__((packed));

// Core-wide header at dm0_isr_blob_offset (before txn threshold + descriptor pools).
// Host ORs txn ids across all DFBs on this core; DM0 loads once for CMDBUF IE programming.
struct dfb_dm0_isr_blob_core_header_t {
    uint32_t producer_txn_id_mask;
    uint32_t consumer_txn_id_mask;
};

// CMDBUF threshold for one txn id (role/path implied by producer/consumer masks in core_hdr).
struct dfb_dm0_isr_txn_threshold_t {
    uint8_t threshold;
    uint8_t _pad;
};

// 32-byte image matching TxnDFBDescriptor layout in dataflow_buffer_interface.h.
struct dfb_dm0_txn_descriptor_image_t {
    uint8_t num_counters;
    uint8_t tile_counters[18];
    uint8_t tiles_to_post_or_ack;  // union post/ack share offset in TxnDFBDescriptor
    uint8_t _pad[12];
};

// One entry per producer RISC that uses the remapper.
// 16 bytes (power of 2): array index compiles to i << 4 (shift, not multiply).
//
// clientR_val / clientL_val are pre-computed on the host using the same bitfield layout
// as tClientR_Config_Reg_u / tClientL_Config_Reg_u in remapper_common.hpp:
//
//   clientR_val  [31:0]:
//     slot r occupies bits [r*8+7 : r*8]:  id_r[2:0] at bit r*8, cnt_sel_r[4:0] at bit r*8+3
//     for each consumer r: clientR_val |= (id_R & 0x7) << (r*8) | (tc_R & 0x1F) << (r*8 + 3)
//
//   clientL_val  [31:0]:
//     [2:0]  = producer_client_type (id_L)
//     [7:3]  = tc_id (cnt_sel_L)
//     [11:8] = (1 << num_clientRs) - 1  (valid mask)
//     [12]   = 1  (clientl_is_producer, always 1 for DFB producers)
//     [13]   = 1  (clientr_group, always 1 for DFB fan-out)
//     [14]   = 0  (distribute, always 0)
//
// Device side: setup_dfb_remapper() writes clientR_val/clientL_val directly to remapper HW
// registers (no staging through g_remapper_configurator arrays).
struct dfb_dm0_remapper_slot_t {
    uint8_t  pair_index;          // remapper pair index for this producer
    // The following three are only consumed when DFB_DM1_TC_INIT_OPTION != 0, where DM1 resets/
    // sets-capacity for the producer's tile counter and publishes readiness on its behalf after
    // enabling the remapper. They occupy former padding bytes (struct stays 16 bytes).
    uint8_t  packed_tile_counter; // producer's tile counter (tensix_id<<5 | tc_id)
    uint8_t  capacity;            // producer TC capacity in entries
    uint8_t  producer_signal_bit; // bit in dfb_signal[logical_dfb_id]; 0xFF = no slot / skip
    uint32_t clientR_val;  // pre-computed ClientR config register value
    uint32_t clientL_val;  // pre-computed ClientL config register value
    uint32_t _pad2;        // pad to 16 bytes
} __attribute__((packed));
static_assert(sizeof(dfb_dm0_remapper_slot_t) == 16, "dfb_dm0_remapper_slot_t must be 16 bytes");

static_assert(sizeof(dfb_dm0_isr_blob_core_header_t) == 8, "dfb_dm0_isr_blob_core_header_t must be 8 bytes");
static_assert(sizeof(dfb_dm0_txn_descriptor_image_t) == 32, "dfb_dm0_txn_descriptor_image_t must be 32 bytes");
static_assert(sizeof(dfb_dm0_isr_txn_threshold_t) == 2, "dfb_dm0_isr_txn_threshold_t must be 2 bytes");

// Span covers txn ids 0 .. highest set bit in (producer_mask | consumer_mask).
inline uint32_t dm0_isr_txn_slot_span(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    const uint32_t all_mask = producer_txn_id_mask | consumer_txn_id_mask;
    if (all_mask == 0) {
        return 0;
    }
    return 32u - static_cast<uint32_t>(__builtin_clz(all_mask));
}

inline uint32_t dm0_isr_txn_hw_pool_byte_size(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return dm0_isr_txn_slot_span(producer_txn_id_mask, consumer_txn_id_mask) * sizeof(dfb_dm0_isr_txn_threshold_t);
}

inline uint32_t dm0_isr_txn_desc_pool_byte_size(uint32_t producer_txn_id_mask, uint32_t consumer_txn_id_mask) {
    return dm0_isr_txn_slot_span(producer_txn_id_mask, consumer_txn_id_mask) * sizeof(dfb_dm0_txn_descriptor_image_t);
}

static_assert(sizeof(dfb_global_header_t) == 96, "dfb_global_header_t size changed — check field alignment");
static_assert(sizeof(dfb_dm1_remapper_entry_header_t) == 4, "dfb_dm1_remapper_entry_header_t must be 4 bytes");
static_assert(sizeof(TCAddressEntry) == 8, "TCAddressEntry size is incorrect");
static_assert(sizeof(dfb_initializer_t) == 32, "dfb_initializer_t size is incorrect");
static_assert(sizeof(dfb_initializer_per_risc_t) == 64, "dfb_initializer_per_risc_t size is incorrect");
static_assert(sizeof(dfb_initializer_intra_tensix_t) == 24, "dfb_initializer_intra_tensix_t size is incorrect");
static_assert(sizeof(dfb_hart_init_entry_t) == 24, "dfb_hart_init_entry_t must be 24B");

namespace dfb {

// ---------------------------------------------------------------------------
// DFB init timing scratch (written by device during setup_*; host reads after benchmarks).
// Layout: 16 fixed slots × 16 uint32 words (64 B each), 1024 B total in cached L1
// (tail of the 4 MiB region so host watcher reads succeed and DFB config is not clobbered).
//
// Slot order:
//   0-7:  DM0-DM7
//   8-15: Neo0 unpack, Neo0 pack, Neo1 unpack, Neo1 pack, Neo2 unpack, Neo2 pack,
//         Neo3 unpack, Neo3 pack
//
// Per-role metrics (A..J = METRIC_A..METRIC_J):
//   DM0_ISR: A=pre_loop_sw B=subpassB_desc C=between_dfb_sw D=subpassB_l1_read
//            E=subpassB_rocc_issue F=first_ie_rmw G=second_ie_rmw H=isr_enable
//            I=implicit_sync_stores J=subpassB_hw
//   DM1_RMP: A=blob_l1_read_sw B=blob_loop_ovhd C=pairs_reg_hw D=enable_remapper_hw
//            E=first_pair_clientR_hw F=first_pair_clientL_hw G=last_pair_hw
//            J=pairs_slots_written
//   DM_LOCAL/TRISC: A=merged_sw B=remapper_spin C=tc_hw D=wait_all E=tc_reset_hw
//                   F=tc_capacity_hw
// ---------------------------------------------------------------------------
constexpr uint8_t DFB_INIT_TIMING_NUM_SLOTS = 16;
constexpr uint8_t DFB_INIT_TIMING_WORDS_PER_SLOT = 16;
constexpr uint32_t DFB_INIT_TIMING_REGION_BYTES =
    static_cast<uint32_t>(DFB_INIT_TIMING_NUM_SLOTS) * static_cast<uint32_t>(DFB_INIT_TIMING_WORDS_PER_SLOT) *
    sizeof(uint32_t);
// Cached L1 byte offset for host reads (DEBUG_VALID_L1_ADDR allows [0, 4 MiB)).
// Device writes use MEM_L1_UNCACHED_BASE + this offset so TL1 is updated without L2 flush.
constexpr uint32_t DFB_INIT_TIMING_L1_BYTE_OFFSET =
    (4u * 1024u * 1024u) - DFB_INIT_TIMING_REGION_BYTES;

constexpr uint32_t DFB_INIT_TIMING_MAGIC = 0xDFB07100u;

enum DfbInitTimingRole : uint8_t {
    DFB_INIT_TIMING_ROLE_DM0_ISR = 0,
    DFB_INIT_TIMING_ROLE_DM1_RMP = 1,
    DFB_INIT_TIMING_ROLE_DM_LOCAL = 2,
    DFB_INIT_TIMING_ROLE_TRISC_LOCAL = 3,
};

enum DfbInitTimingWord : uint8_t {
    DFB_INIT_TIMING_W_MAGIC = 0,
    DFB_INIT_TIMING_W_VALID = 1,
    DFB_INIT_TIMING_W_ROLE = 2,
    DFB_INIT_TIMING_W_E2E = 3,
    DFB_INIT_TIMING_W_METRIC_A = 4,
    DFB_INIT_TIMING_W_METRIC_B = 5,
    DFB_INIT_TIMING_W_METRIC_C = 6,
    DFB_INIT_TIMING_W_METRIC_D = 7,
    DFB_INIT_TIMING_W_METRIC_E = 8,
    DFB_INIT_TIMING_W_METRIC_F = 9,
    DFB_INIT_TIMING_W_START = 10,
    DFB_INIT_TIMING_W_END = 11,
    DFB_INIT_TIMING_W_METRIC_G = 12,
    DFB_INIT_TIMING_W_METRIC_H = 13,
    DFB_INIT_TIMING_W_METRIC_I = 14,
    DFB_INIT_TIMING_W_METRIC_J = 15,
};

}  // namespace dfb
