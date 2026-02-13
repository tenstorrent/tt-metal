// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace experimental {

enum AccessPattern : uint8_t {  // this should be put into experimental/hostdev or should it be a host file???
    STRIDED,
    BLOCKED,
    UNKNOWN,
};

constexpr uint8_t NUM_TENSIX = 4;
constexpr uint8_t NUM_TILE_COUNTERS_PER_TENSIX = 32;
constexpr uint8_t NUM_TENSIX_TILE_COUNTERS_FOR_DM = 16;
constexpr uint8_t NUM_REMAPPER_PAIRINGS = 64;
constexpr uint8_t NUM_TXN_IDS = 4;
constexpr uint8_t MAX_NUM_TILE_COUNTERS_TO_RR = 4;

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

// move configs and LocalDFBInterface structs to hw/inc/hostdev

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
    (24 + (44 * 12)) * 16 = 8320 bytes
*/
struct dfb_initializer_t {  // 24 bytes
    uint32_t logical_id;
    uint32_t entry_size;
    uint32_t stride_size;
    uint16_t capacity;
    struct {
        uint16_t dm_mask : 8;         // bits 0-7: DM RISC mask
        uint16_t tensix_mask : 4;     // bits 8-11: Tensix RISC mask
        uint16_t reserved : 3;        // bits 12-14: unused
        uint16_t tc_initialized : 1;  // bit 15: tile counter initialized flag
    } risc_mask_bits;
    uint8_t num_txn_ids;
    uint8_t txn_ids[NUM_TXN_IDS];
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t padding;  // Reserved for future use (remapper_consumer_ids_mask moved to per-RISC)
} __attribute__((packed));

struct dfb_initializer_per_risc_t {  // 44 bytes
    uint32_t base_addr[MAX_NUM_TILE_COUNTERS_TO_RR];
    uint32_t limit[MAX_NUM_TILE_COUNTERS_TO_RR];
    PackedTileCounter packed_tile_counter[MAX_NUM_TILE_COUNTERS_TO_RR];
    uint8_t num_tcs_to_rr;
    struct {
        uint8_t remapper_pair_index : 6;  // bits 0-5: 0..63
        uint8_t remapper_en : 1;          // bit 6
        uint8_t should_init_tc : 1;  // bit 7: 1 = this RISC should initialize tile counters and program the remapper
    } __attribute__((packed)) flags;
    uint32_t consumer_tcs;  // used to program remapper, for a L:R mapping contains all the TCs on the consumer side
                            // (R). TC can be value between 0 and 31 (5 bits, max of 4 TCs)
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
    PackedTileCounter packed_tile_counter;
    uint8_t tensix_mask;
} __attribute__((packed));

// on WH/BH arrays will be sized to 1
struct LocalDFBInterface {
    uint32_t rd_ptr[MAX_NUM_TILE_COUNTERS_TO_RR];
    uint32_t wr_ptr[MAX_NUM_TILE_COUNTERS_TO_RR];
    uint32_t base_addr[MAX_NUM_TILE_COUNTERS_TO_RR];
    uint32_t limit[MAX_NUM_TILE_COUNTERS_TO_RR];

    uint32_t entry_size;   // shared across riscs so can be factored out and put into sep initialization struct
    uint32_t stride_size;  // shared across riscs so can be factored out and put into sep initialization struct

    PackedTileCounter packed_tile_counter[MAX_NUM_TILE_COUNTERS_TO_RR];
    uint8_t txn_ids[NUM_TXN_IDS];  // shared across riscs so can be factored out and put into sep initialization struct
    uint8_t
        num_entries_per_txn_id;  // shared across riscs so can be factored out and put into sep initialization struct
    uint8_t num_entries_per_txn_id_per_tc;  // shared across riscs so can be factored out and put into sep
                                            // initialization struct
    uint8_t remapper_pair_index;
    uint8_t num_tcs_to_rr;
    uint8_t num_txn_ids;  // shared across riscs so can be factored out and put into sep initialization struct

    uint8_t padding[3];

    // #ifndef ARCH_QUASAR
    //     // used by packer for in-order packing ... is this still needed on Quasar?
    //     uint32_t wr_tile_ptr;

    //     // Save a cycle during init by writing 0 to the uint32 below
    //     union {
    //         uint32_t tiles_acked_received_init;
    //         struct {
    //             uint16_t tiles_acked;
    //             uint16_t tiles_received;
    //         };
    //     };
    // #endif
} __attribute__((packed));

static_assert(sizeof(dfb_initializer_t) == 24, "dfb_initializer_t size is incorrect");
static_assert(sizeof(dfb_initializer_per_risc_t) == 44, "dfb_initializer_per_risc_t size is incorrect");
static_assert(sizeof(dfb_initializer_intra_tensix_t) == 24, "dfb_initializer_intra_tensix_t size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 88, "LocalDFBInterface size is incorrect");

}  // namespace experimental
