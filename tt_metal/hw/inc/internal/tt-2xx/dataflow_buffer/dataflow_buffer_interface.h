// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include "dataflow_buffer_config.h"
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/remapper_api.hpp"
#endif

// Forward declarations
struct LocalDFBInterface;
struct TxnDFBDescriptor;

// Global DFB interface array
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
extern thread_local LocalDFBInterface g_dfb_interface[dfb::MAX_ACTIVE_DFBS_PACK];
extern thread_local uint8_t g_dfb_logical_to_compact[dfb::NUM_DFBS];
#else
extern thread_local LocalDFBInterface g_dfb_interface[dfb::NUM_DFBS];
#endif

// Cached L1 byte address of the DFB config region; set once during setup_local_dfb_interfaces
// so DataflowBuffer::DataflowBuffer() can call dfb_ensure_ready without a separate parameter.
extern thread_local uintptr_t g_dfb_config_base_addr;
#ifndef COMPILE_FOR_TRISC
extern volatile TxnDFBDescriptor g_txn_dfb_descriptor[32];
extern overlay::RemapperAPI g_remapper_configurator;
#endif

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)

struct DFBTCSlot {
    uint32_t base_addr;
    uint16_t wr_offset;
    uint16_t ring_size;
    uint16_t wr_entry_idx;
    uint16_t base_entry_idx;
    dfb::PackedTileCounter packed_tile_counter;
} __attribute__((packed));

struct LocalDFBInterface {
    uint16_t entry_size;
    uint16_t stride_size;
    uint16_t wr_entry_ptr;
    uint8_t stride_size_tiles;
    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;
    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
} __attribute__((packed));

static_assert(sizeof(DFBTCSlot) == 13, "DFBTCSlot (pack TRISC) size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 87, "LocalDFBInterface (pack TRISC) size is incorrect");

#elif defined(COMPILE_FOR_TRISC)

struct DFBTCSlot {
    uint32_t base_addr;
    uint16_t rd_offset;
    uint16_t ring_size;
    uint16_t rd_entry_idx;
    uint16_t base_entry_idx;
    dfb::PackedTileCounter packed_tile_counter;
} __attribute__((packed));

struct LocalDFBInterface {
    uint16_t entry_size;
    uint16_t stride_size;
    uint8_t stride_size_tiles;
    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;
    uint8_t tensix_trisc_mask;
    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
} __attribute__((packed));

static_assert(sizeof(DFBTCSlot) == 13, "DFBTCSlot (unpack TRISC) size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 86, "LocalDFBInterface (unpack TRISC) size is incorrect");

#else

// Per–tile-counter slot (DM).
//
// Layout is designed so that every DFBTCSlot in the tc_slots[] array is naturally
// 4B-aligned, eliminating misaligned-store penalties in setup_local_dfb_interfaces:
//
//   - LocalDFBInterface has a 20B scalar prefix (19 bytes of fields + 1B _tc_align_pad),
//     so tc_slots[0] starts at offset 20 (divisible by 4).
//   - DFBTCSlot is padded to 20B (next multiple of 4 after 17B), so tc_slots[i] starts
//     at offset 20 + i*20 which is always divisible by 4.
//   - sizeof(LocalDFBInterface) = 20 + 6*20 = 140 = 4*35, so g_dfb_interface[id] is
//     4B-aligned for every id when the array base is 4B-aligned.
//
// Without alignment: rd_ptr/wr_ptr/base_addr/limit would sit at misaligned offsets,
// causing the compiler to emit 4 byte-stores per u32 field (16 instead of 4 stores
// per slot per DFB init entry).
struct DFBTCSlot {
    uint32_t rd_ptr;
    uint32_t wr_ptr;
    uint32_t base_addr;
    uint32_t limit;
    dfb::PackedTileCounter packed_tile_counter;
    uint8_t _align[3];  // pad 17 → 20B so every slot in tc_slots[] is 4B-aligned
} __attribute__((packed));

// on WH/BH arrays will be sized to 1
struct LocalDFBInterface {
    uint32_t entry_size;
    uint32_t stride_size;

    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;

    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t threshold;        // When this value is met, ISR to post/ack credits will fire.
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t num_txn_ids;
    uint8_t broadcast_tc;  // DM-DM ALL producer: post to all TCs instead of round-robin
    uint8_t _tc_align_pad;  // pad 19B prefix → 20B so tc_slots[0] is 4B-aligned

    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
} __attribute__((packed));

static_assert(sizeof(DFBTCSlot) == 20, "DFBTCSlot size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 140, "LocalDFBInterface size is incorrect");

#endif

inline LocalDFBInterface& get_local_dfb_interface(uint32_t logical_dfb_id) {
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    return g_dfb_interface[g_dfb_logical_to_compact[logical_dfb_id]];
#else
    return g_dfb_interface[logical_dfb_id];
#endif
}

// Holds metadata for transaction ID based ISR handling.
// It is used by the ISR to understand which tile counters need to update which credits (post/ack).
// Padded to 32 bytes so g_txn_dfb_descriptor[trid] compiles to base + (trid << 5) instead of a multiply-by-20.
struct TxnDFBDescriptor {
    uint8_t num_counters;
    dfb::PackedTileCounter tile_counters[18];
    union {
        uint8_t tiles_to_post;
        uint8_t tiles_to_ack;
    } __attribute__((packed));
    uint8_t _pad[12];  // pad 20 → 32 bytes
};
static_assert(sizeof(TxnDFBDescriptor) == 32, "TxnDFBDescriptor size is incorrect");
static_assert(
    sizeof(TxnDFBDescriptor) == sizeof(dfb_dm0_txn_descriptor_image_t),
    "TxnDFBDescriptor must match dfb_dm0_txn_descriptor_image_t for ISR blob memcpy");
static_assert(
    offsetof(TxnDFBDescriptor, num_counters) == offsetof(dfb_dm0_txn_descriptor_image_t, num_counters),
    "TxnDFBDescriptor field layout must match dfb_dm0_txn_descriptor_image_t");
static_assert(
    offsetof(TxnDFBDescriptor, tile_counters) == offsetof(dfb_dm0_txn_descriptor_image_t, tile_counters),
    "TxnDFBDescriptor field layout must match dfb_dm0_txn_descriptor_image_t");
static_assert(
    offsetof(TxnDFBDescriptor, tiles_to_post) == offsetof(dfb_dm0_txn_descriptor_image_t, tiles_to_post_or_ack),
    "TxnDFBDescriptor tiles_to_post must match dfb_dm0_txn_descriptor_image_t tiles_to_post_or_ack");
