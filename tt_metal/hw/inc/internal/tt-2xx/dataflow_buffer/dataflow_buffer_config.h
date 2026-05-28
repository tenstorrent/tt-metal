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
      dfb_global_header_t (4B)  — stores per_dfb_layout_offset
      DM0 global blob           — contiguous across all DFBs:
        [DFB0: dfb_dm0_blob_header_t + rmp_slots + txn_entries]
        [DFB1: dfb_dm0_blob_header_t + rmp_slots + txn_entries]
        ...
    [dfb_config_base + per_dfb_layout_offset]:
      DFB 0: [dfb_initializer_t (32B)] [dfb_initializer_per_risc_t (64B) × N]
      DFB 1: [dfb_initializer_t (32B)] [dfb_initializer_per_risc_t (64B) × N]
      ...

    DM0 reads linearly from dfb_config_base + sizeof(dfb_global_header_t) — no
    dfb_initializer_t fetches, no per-RISC cache pollution, hardware prefetcher-friendly.
    Other RISCs read from dfb_config_base + per_dfb_layout_offset — tighter stride,
    no dm0_blob_size field to skip.

    Base cost (1Sx1S, 2 riscs, 1 txn blob):
      4 + 20 + (32 + 64*2) * 1 = 184 bytes
    Worst case (4Sx4A, 5 riscs, 4 rmp slots, 1 txn), 8 DFBs:
      4 + 84*8 + (32 + 64*5)*8 = 4 + 672 + 2816 = 3492 bytes
*/

// Single 4-byte header at the start of the DFB config region.
// Tells all RISCs (and DM0) where the shared per-DFB layout begins.
struct dfb_global_header_t {
    uint32_t per_dfb_layout_offset;  // byte offset from dfb_config_base to first dfb_initializer_t
} __attribute__((packed));
struct dfb_txn_id_descriptor_t {
    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t num_entries_to_process_threshold; // entries each txn ID tracks before posting/acking
    uint8_t num_txn_ids;
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
} __attribute__((packed));

struct dfb_initializer_t {  // 32 bytes
    uint32_t entry_size;
    uint32_t stride_in_entries;
    uint16_t capacity;
    struct {
        uint16_t dm_mask : 8;         // bits 0-7: DM RISC mask
        uint16_t tensix_mask : 4;     // bits 8-11: Neo RISC mask
        uint16_t tensix_trisc_mask : 4;        // bits 12-15: indicates which triscs use the DFB (tensix producer uses trisc2 and tensix consumer can use trisc0 or trisc3)
    } risc_mask_bits;
    // For DM-to-DM DFBs, producer and consumer would have different set of transaction ids
    dfb_txn_id_descriptor_t producer_txn_descriptor;
    dfb_txn_id_descriptor_t consumer_txn_descriptor;
    uint8_t num_producers;
    uint8_t implicit_sync_configured; // 0: init state, 1: configured
    uint8_t _pad[2];                  // reserved (was dm0_blob_size; DM0 blob is now a separate global region)
} __attribute__((packed));

// AoS layout: base_addr and limit for the same TC slot are adjacent in memory,
// improving spatial locality in the TC-init loop that accesses both per iteration.
struct TCAddressEntry {
    uint32_t base_addr;
    uint32_t limit;
} __attribute__((packed));

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
} __attribute__((packed));

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

// Host-pre-computed blob appended after the per-risc entries for each DFB.
// DM0 reads this directly in subpassB, eliminating the risc-scan loop and bit-manipulation overhead.
// Variable-length: only the needed entries are serialized. The count is stored in dfb_dm0_blob_header_t.

struct dfb_dm0_blob_header_t {  // 4 bytes
    uint8_t num_remapper_slots;
    uint8_t num_producer_txns;
    uint8_t num_consumer_txns;
    uint8_t _pad;
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

static_assert(sizeof(dfb_global_header_t) == 4, "dfb_global_header_t must be 4 bytes");
static_assert(sizeof(TCAddressEntry) == 8, "TCAddressEntry size is incorrect");
static_assert(sizeof(dfb_initializer_t) == 32, "dfb_initializer_t size is incorrect");
static_assert(sizeof(dfb_initializer_per_risc_t) == 64, "dfb_initializer_per_risc_t size is incorrect");
static_assert(sizeof(dfb_initializer_intra_tensix_t) == 24, "dfb_initializer_intra_tensix_t size is incorrect");
