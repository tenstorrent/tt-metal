// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace dfb {

enum AccessPattern : uint8_t {
    STRIDED,
    BLOCKED,
    UNKNOWN,
};

constexpr uint8_t NUM_DFBS = 32;

constexpr uint8_t NUM_TENSIX = 4;
constexpr uint8_t NUM_TILE_COUNTERS_PER_TENSIX = 32;
constexpr uint8_t NUM_TENSIX_TILE_COUNTERS_FOR_DM = 16;
constexpr uint8_t NUM_REMAPPER_PAIRINGS = 64;
constexpr uint8_t NUM_TXN_IDS = 4;
constexpr uint8_t MAX_NUM_TILE_COUNTERS_TO_RR = 4;

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
    tensix + dm or tensix + tensix via remapper dfb
    LogicalDFB 0:
    | dfb_initializer_t |
    | dfb_initializer_per_risc_t | risc 0
    | dfb_initializer_per_risc_t | risc 1
    ...
    | dfb_initializer_per_risc_t | risc 11
    LogicalDFB 1:
    | dfb_initializer_t |
    | dfb_initializer_per_risc_t | risc 0
    | dfb_initializer_per_risc_t | risc 1
    ...
    (36 + (44 * 12)) * 16 = 8336 bytes
*/
struct dfb_txn_id_descriptor_t {
    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t num_entries_to_process_threshold; // entries each txn ID tracks before posting/acking
    uint8_t num_txn_ids;
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
} __attribute__((packed));

struct dfb_initializer_t {  // 36 bytes
    uint32_t logical_id;
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
    uint8_t padding[3];
} __attribute__((packed));

struct dfb_initializer_per_risc_t {  // 44 bytes
    uint32_t base_addr[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    uint32_t limit[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    uint32_t consumer_tcs;  // used to program remapper, for a L:R mapping contains all the TCs on the consumer side
                            // (R). TC can be value between 0 and 31 (5 bits, max of 4 TCs)
    dfb::PackedTileCounter packed_tile_counter[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    struct {
        uint8_t num_tcs_to_rr : 4;   // 0..8, number of TCs to round-robin (max 4 but keeping space)
        uint8_t tc_init_done : 1;
        uint8_t broadcast_tc : 1;    // DM-DM BLOCKED: producer posts to all TCs instead of round-robin
        uint8_t reserved : 2;
    } __attribute__((packed)) num_tcs_and_init;
    struct {
        uint8_t remapper_pair_index : 6;  // bits 0-5: 0..63
        uint8_t remapper_en : 1;          // bit 6
        uint8_t is_producer : 1;  // bit 7: indicates if this RISC is a producer
    } __attribute__((packed)) flags;
    uint8_t remapper_consumer_ids_mask;  // Bitmask of clientTypes (id_R) for this producer's consumers
    uint8_t producer_client_type;        // clientL for this producer when using remapper
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

static_assert(sizeof(dfb_initializer_t) == 36, "dfb_initializer_t size is incorrect");
static_assert(sizeof(dfb_initializer_per_risc_t) == 44, "dfb_initializer_per_risc_t size is incorrect");
static_assert(sizeof(dfb_initializer_intra_tensix_t) == 24, "dfb_initializer_intra_tensix_t size is incorrect");
