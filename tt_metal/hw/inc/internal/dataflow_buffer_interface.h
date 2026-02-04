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

using PackedTileCounter = uint8_t;  // top 2 bits identify tensix id, bottom 5 bits for counter id

// NOLINTBEGIN(readability-redundant-inline-specifier)
inline __attribute__((always_inline)) constexpr uint8_t get_tensix_id(PackedTileCounter p) { return (p >> 5) & 0x03; }

inline __attribute__((always_inline)) constexpr uint8_t get_counter_id(PackedTileCounter p) { return p & 0x1F; }
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
    uint8_t txn_ids[4];
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t remapper_consumer_mask;  // used to program remapper, for a L:R mapping, indicates which riscs make up R
} __attribute__((packed));

struct dfb_initializer_per_risc_t {  // 44 bytes
    uint32_t base_addr[4];
    uint32_t limit[4];
    PackedTileCounter packed_tile_counter[4];
    uint8_t num_tcs_to_rr;
    struct {
        uint8_t remapper_pair_index : 6;  // bits 0-5: 0..63
        uint8_t remapper_en : 1;          // bit 6
        uint8_t should_init_tc : 1;  // bit 7: 1 = this RISC should initialize tile counters and program the remapper
    } __attribute__((packed)) flags;
    uint32_t consumer_tcs;  // used to program remapper, for a L:R mapping contains all the TCs on the consumer side
                            // (R). TC can be value between 0 and 31
    uint8_t padding[2];
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
    uint32_t rd_ptr[4];
    uint32_t wr_ptr[4];
    uint32_t base_addr[4];
    uint32_t limit[4];

    uint32_t entry_size;   // shared across riscs so can be factored out and put into sep initialization struct
    uint32_t stride_size;  // shared across riscs so can be factored out and put into sep initialization struct

    PackedTileCounter packed_tile_counter[4];
    uint8_t txn_ids[4];  // shared across riscs so can be factored out and put into sep initialization struct
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

}  // namespace experimental
