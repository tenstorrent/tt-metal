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
extern TxnDFBDescriptor g_txn_dfb_descriptor[32];
extern overlay::RemapperAPI g_remapper_configurator;
#endif

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)

// ring_size is uint32 L1-aligned units so a DFB can span full Quasar L1 (~4 MB).
// Cursor byte-offset is not stored: it is derived from wr_entry_idx (advances in lockstep
// with stride_size / stride_size_tiles).
//
// 16B and 4B-aligned rather than a packed 13B: in the packed form every slot after the first starts
// at an odd offset, so GCC writes each field one byte at a time.
struct alignas(4) DFBTCSlot {
    uint32_t base_addr;
    uint32_t ring_size;
    uint16_t wr_entry_idx;
    uint16_t base_entry_idx;
    dfb::PackedTileCounter packed_tile_counter;
    uint8_t _pad[3];
};

extern thread_local DFBTCSlot g_dfb_tc_slots[dfb::MAX_PACK_TC_SLOTS];

// Preserve the iface.tc_slots[i] API while allocating only the slots each active Pack DFB needs.
// slot_base indexes that DFB's contiguous slice of g_dfb_tc_slots.
struct DFBTCSlotSpan {
    uint8_t slot_base;

    inline __attribute__((always_inline)) DFBTCSlot& operator[](uint32_t index) {
        return g_dfb_tc_slots[slot_base + index];
    }

    inline __attribute__((always_inline)) const DFBTCSlot& operator[](uint32_t index) const {
        return g_dfb_tc_slots[slot_base + index];
    }
} __attribute__((packed));

struct LocalDFBInterface {
    uint16_t entry_size;
    uint16_t stride_size;
    uint16_t num_entries;
    uint16_t wr_entry_ptr;
    uint16_t block_size;  // entries in one block; 1 unless the ring is BLOCKED
    uint16_t split_tc;    // one op (a whole block) is split across all of this hart's TCs
    uint16_t jump;        // cursor jump in entries, taken on each TC rotation
    uint8_t stride_size_tiles;
    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;
    DFBTCSlotSpan tc_slots;
} __attribute__((packed, aligned(2)));

static_assert(sizeof(DFBTCSlot) == 16, "DFBTCSlot (pack TRISC) size is incorrect");
static_assert(sizeof(DFBTCSlotSpan) == 1, "DFBTCSlotSpan (pack TRISC) size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 18, "LocalDFBInterface (pack TRISC) size is incorrect");

#elif defined(COMPILE_FOR_TRISC)

// Same layout as pack: uint32 ring_size, cursor offset derived from rd_entry_idx, 16B and 4B-aligned.
struct alignas(4) DFBTCSlot {
    uint32_t base_addr;
    uint32_t ring_size;
    uint16_t rd_entry_idx;
    uint16_t base_entry_idx;
    dfb::PackedTileCounter packed_tile_counter;
    uint8_t _pad[3];
};

struct LocalDFBInterface {
    uint16_t entry_size;
    uint16_t stride_size;
    uint16_t num_entries;
    uint8_t stride_size_tiles;
    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;
    uint8_t tensix_trisc_mask;
    uint16_t block_size;
    uint16_t split_tc;
    uint16_t jump;
    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
};

static_assert(sizeof(DFBTCSlot) == 16, "DFBTCSlot (unpack TRISC) size is incorrect");
static_assert(offsetof(LocalDFBInterface, tc_slots) == 16, "tc_slots (unpack TRISC) must start at 16");
static_assert(sizeof(LocalDFBInterface) == 112, "LocalDFBInterface (unpack TRISC) size is incorrect");

#else

// Per–tile-counter slot (DM). A DM hart is either producer or consumer of a DFB, never both, so
// one cursor (`ptr`) serves as its write or read pointer.
struct DFBTCSlot {
    uint32_t ptr;
    uint32_t base_addr;
    uint32_t limit;
    dfb::PackedTileCounter packed_tile_counter;
    uint8_t _align[3];  // pad 13 → 16B so every slot in tc_slots[] is 4B-aligned
};

// alignas(64): g_dfb_interface is thread_local and only 8B-aligned otherwise, so a slot could start
// mid-line and the init copy of its first 48 bytes would touch two cache lines instead of one.
// sizeof is unchanged: 128 is already a multiple of 64. The host builds this layout byte for byte
// as the DM init image (dfb_write_dm_image_entry in dataflow_buffer_config.h).
struct alignas(64) LocalDFBInterface {
    uint32_t entry_size;
    uint32_t stride_size;

    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;

    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t threshold;  // When this value is met, ISR to post/ack credits will fire.
    uint8_t num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t num_txn_ids;
    uint8_t broadcast_tc;   // DM-DM ALL producer: post to all TCs instead of round-robin
    uint8_t _tc_align_pad;  // pad bytes [8,20) → 20B so tc_slots[] stays 4B-aligned

    uint16_t num_entries;
    uint16_t block_size;  // entries in one block; 1 unless the ring is BLOCKED
    uint16_t split_tc;    // one op (a whole block) belongs to every one of this hart's TCs,
                          // so each TC gets its share of the credits instead of one TC
                          // everything: Blocked->Strided producers, Strided->Blocked consumers
    uint32_t jump;        // cursor jump in bytes, taken on each TC rotation

    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
};

static_assert(sizeof(DFBTCSlot) == 16, "DFBTCSlot size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 128, "LocalDFBInterface size is incorrect");
static_assert(offsetof(LocalDFBInterface, tc_slots) == 32, "tc_slots must start at 32");

#endif

#if defined(COMPILE_FOR_TRISC)
// Byte-offset (L1 units) of the current FIFO cursor from slot.base_addr.
// wr/rd_entry_idx and the former wr/rd_offset advance in lockstep:
//   entry += n * stride_size_tiles;  offset += n * stride_size;
// so offset is recoverable without storing it (saves 4B/slot on pack TLS).
inline __attribute__((always_inline)) uint32_t
dfb_slot_cursor_offset_units(const LocalDFBInterface& intf, const DFBTCSlot& slot, uint32_t entry_idx) {
    const uint32_t delta_entries = entry_idx - static_cast<uint32_t>(slot.base_entry_idx);
    return (delta_entries / static_cast<uint32_t>(intf.stride_size_tiles)) * static_cast<uint32_t>(intf.stride_size);
}
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
// Hot fields first: num_counters and tiles_to_post/ack are bytes 0-1, so the ISR's first load covers
// both. The offsetof asserts below keep it identical to the host image. alignas(32): each descriptor
// is one aligned half-line, which the 64-bit init copy relies on.
struct alignas(32) TxnDFBDescriptor {
    uint8_t num_counters;
    union {
        uint8_t tiles_to_post;
        uint8_t tiles_to_ack;
    } __attribute__((packed));
    dfb::PackedTileCounter tile_counters[dfb::MAX_TILE_COUNTERS_PER_SIDE];
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
