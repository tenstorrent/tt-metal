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
      dfb_global_header_t (64B)
        + uint16_t dfb_byte_offset[num_dfbs]
        + uint16_t per_risc_byte_offset[num_dfbs][NUM_PARTICIPATING_HARTIDS]
      — variable-length prefix (4-byte padded)

    [dfb_config_base + dm1_remapper_blob_offset]:
      DM1 remapper blob — contiguous across all DFBs, read by DM1:
        [DFB0: dfb_dm1_remapper_entry_header_t(4B) + dfb_dm0_remapper_slot_t × n]
        [DFB1: ...]
        ...

    [dfb_config_base + dm0_isr_blob_offset]:
      DM0 ISR blob — contiguous across all DFBs, read by DM0:
        [DFB0: dfb_dm0_isr_entry_header_t(4B) + dfb_dm0_txn_entry_t × (num_prod + num_cons)]
        [DFB1: ...]
        ...

    [dfb_config_base + per_dfb_layout_offset]:
      DFB 0: [dfb_initializer_t (32B)] [dfb_initializer_per_risc_t (64B) × N]
      DFB 1: [dfb_initializer_t (32B)] [dfb_initializer_per_risc_t (64B) × N]
      ...

    DM1 reads linearly through only remapper slot data (cache-efficient, no txn/init pollution).
    DM0 reads linearly through only ISR txn data (cache-efficient, no remapper/init pollution).
    DM1 and DM0 run their respective blob loops in parallel.
    Other DMs/TRISCs read from per_dfb_layout_offset (no DM blob pollution).

    Base cost (1Sx1S, 2 riscs, 1 txn blob):
      66 + 4 + 20 + (32 + 64*2) * 1 = 250 bytes
    Worst case (4Sx4A, 5 riscs, 4 rmp slots, 1 txn), 8 DFBs:
      80 + (4+4*16)*8 + (4+20)*8 + (32 + 64*5)*8 = 80 + 544 + 192 + 2816 = 3632 bytes
*/

// Fixed header at the start of the DFB config region.
// Immediately followed in L1 by:
//   uint16_t dfb_byte_offset[num_dfbs] — byte offset to dfb_initializer_t per logical id
//   uint16_t per_risc_byte_offset[num_dfbs][NUM_PARTICIPATING_HARTIDS] — byte offset to this hart's
//     dfb_initializer_per_risc_t (0 when hart does not participate on that DFB)
struct dfb_global_header_t {
    uint32_t dm1_remapper_blob_offset;  // → DM1 remapper blob (dfb_dm1_remapper_entry_header_t + slots per DFB)
    uint32_t dm0_isr_blob_offset;       // → DM0 ISR blob (dfb_dm0_isr_entry_header_t + txn entries per DFB)
    uint32_t per_dfb_layout_offset;     // → shared per-DFB layout (dfb_initializer_t + per_risc entries)
    uint8_t num_dfbs;                   // DFB count on this core; logical ids 0 .. num_dfbs-1
    uint8_t _pad;
    // participation_mask[h] bit i set → hartid h participates in DFB i (host-derived from risc_mask).
    uint32_t participation_mask[dfb::NUM_PARTICIPATING_HARTIDS];
};

inline uint32_t dfb_byte_offset_table_byte_offset() { return sizeof(dfb_global_header_t); }

inline uint32_t dfb_per_risc_byte_offset_table_byte_offset(uint8_t num_dfbs) {
    return dfb_byte_offset_table_byte_offset() + static_cast<uint32_t>(num_dfbs) * sizeof(uint16_t);
}

inline uint32_t dfb_per_risc_byte_offset_table_index(uint8_t logical_dfb_id, uint8_t hartid) {
    return static_cast<uint32_t>(logical_dfb_id) * static_cast<uint32_t>(dfb::NUM_PARTICIPATING_HARTIDS) +
           static_cast<uint32_t>(hartid);
}

inline uint32_t dfb_config_prefix_size(uint8_t num_dfbs) {
    const uint32_t table_end = dfb_per_risc_byte_offset_table_byte_offset(num_dfbs) +
                               static_cast<uint32_t>(num_dfbs) * static_cast<uint32_t>(dfb::NUM_PARTICIPATING_HARTIDS) *
                                   sizeof(uint16_t);
    // Pad prefix so DM1/DM0 blobs and per-DFB layout start on 4-byte boundaries (L1 u32 access).
    return (table_end + 3u) & ~3u;
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
    uint8_t implicit_sync_configured; // 0: init state, 1: configured
    uint8_t _pad[2];                  // reserved (was dm0_blob_size; DM0 blob is now a separate global region)
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

// Per-DFB header for the DM0 ISR blob.
// DM0 reads linearly through this region, processing only CMDBUF threshold / ISR data.
struct dfb_dm0_isr_entry_header_t {  // 4 bytes
    uint8_t num_producer_txns;
    uint8_t num_consumer_txns;
    uint8_t _pad[2];
} __attribute__((packed));

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
// Device side: load_pair_raw(pair_index, clientR_val, clientL_val) writes both fields
// directly into g_remapper_configurator's internal arrays — no bitfield manipulation needed.
struct dfb_dm0_remapper_slot_t {
    uint8_t  pair_index;   // remapper pair index for this producer
    uint8_t  _pad[3];      // pad to 4-byte alignment
    uint32_t clientR_val;  // pre-computed ClientR config register value
    uint32_t clientL_val;  // pre-computed ClientL config register value
    uint32_t _pad2;        // pad to 16 bytes
} __attribute__((packed));
static_assert(sizeof(dfb_dm0_remapper_slot_t) == 16, "dfb_dm0_remapper_slot_t must be 16 bytes");

// One entry per txn ID in the producer or consumer txn descriptor.
// 16 bytes (power of 2): array index compiles to i << 4 (shift, not multiply).
struct dfb_dm0_txn_entry_t {
    uint8_t txn_id;
    uint8_t tiles_to_post_or_ack;  // num_entries_per_txn_id_per_tc
    uint8_t threshold;             // num_entries_to_process_threshold
    uint8_t num_tcs;               // number of valid tile_counters entries
    dfb::PackedTileCounter tile_counters[dfb::MAX_TCS_PER_TXN];  // 8 bytes
    uint8_t _pad[4];               // pad to 16 bytes
} __attribute__((packed));
static_assert(sizeof(dfb_dm0_txn_entry_t) == 16, "dfb_dm0_txn_entry_t must be 16 bytes");

static_assert(sizeof(dfb_global_header_t) == 64, "dfb_global_header_t size changed — check field alignment");
static_assert(sizeof(dfb_dm1_remapper_entry_header_t) == 4, "dfb_dm1_remapper_entry_header_t must be 4 bytes");
static_assert(sizeof(dfb_dm0_isr_entry_header_t) == 4, "dfb_dm0_isr_entry_header_t must be 4 bytes");
static_assert(sizeof(TCAddressEntry) == 8, "TCAddressEntry size is incorrect");
static_assert(sizeof(dfb_initializer_t) == 32, "dfb_initializer_t size is incorrect");
static_assert(sizeof(dfb_initializer_per_risc_t) == 64, "dfb_initializer_per_risc_t size is incorrect");
static_assert(sizeof(dfb_initializer_intra_tensix_t) == 24, "dfb_initializer_intra_tensix_t size is incorrect");
