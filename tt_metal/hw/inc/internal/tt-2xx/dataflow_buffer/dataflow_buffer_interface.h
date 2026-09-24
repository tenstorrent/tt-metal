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

// ring_size is uint32 L1-aligned units so a DFB can span full Quasar L1 (~4 MB).
// Cursor byte-offset is not stored: it is derived from wr_entry_idx (advances in lockstep
// with stride_size / stride_size_tiles). That keeps sizeof(LocalDFBInterface)==89 so
// g_dfb_interface[16] + logical map fit in pack TLS (2048) with the required 256B stack.
struct DFBTCSlot {
    uint32_t base_addr;
    uint32_t ring_size;
    uint16_t wr_entry_idx;
    uint16_t base_entry_idx;
    dfb::PackedTileCounter packed_tile_counter;
} __attribute__((packed));

struct LocalDFBInterface {
    uint16_t entry_size;
    uint16_t stride_size;
    uint16_t num_entries;
    uint16_t wr_entry_ptr;
    uint8_t stride_size_tiles;
    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;
    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
} __attribute__((packed));

static_assert(sizeof(DFBTCSlot) == 13, "DFBTCSlot (pack TRISC) size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 89, "LocalDFBInterface (pack TRISC) size is incorrect");
inline dfb::PackedTileCounter dfb_iface_ptc(const LocalDFBInterface& i, uint32_t t) {
    return i.tc_slots[t].packed_tile_counter;
}

#elif defined(COMPILE_FOR_TRISC)

// Same compact layout as pack: uint32 ring_size, cursor offset derived from rd_entry_idx.
struct DFBTCSlot {
    uint32_t base_addr;
    uint32_t ring_size;
    uint16_t rd_entry_idx;
    uint16_t base_entry_idx;
    dfb::PackedTileCounter packed_tile_counter;
} __attribute__((packed));

struct LocalDFBInterface {
    uint16_t entry_size;
    uint16_t stride_size;
    uint16_t num_entries;
    uint8_t stride_size_tiles;
    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;
    uint8_t tensix_trisc_mask;
    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
} __attribute__((packed));

static_assert(sizeof(DFBTCSlot) == 13, "DFBTCSlot (unpack TRISC) size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 88, "LocalDFBInterface (unpack TRISC) size is incorrect");
inline dfb::PackedTileCounter dfb_iface_ptc(const LocalDFBInterface& i, uint32_t t) {
    return i.tc_slots[t].packed_tile_counter;
}

#else

// Per–tile-counter slot (DM).
//
// Exactly 16B with no padding. packed_tile_counter used to live here, which forced the struct to
// pad 17 -> 20 and cost that padding once per slot (6x). It now lives in LocalDFBInterface::ptc[],
// where the six bytes are contiguous. 16 is a power of two, so slot indexing is a shift.
struct DFBTCSlot {
    uint32_t rd_ptr;
    uint32_t wr_ptr;
    uint32_t base_addr;
    uint32_t limit;
};

// alignas(64), not 8: g_dfb_interface sits at TLS offset 0x828, and 0x828 % 64 == 40, so with
// only 8B alignment every 128B slot starts 40 bytes into a cache line. The init copy writes the
// first 48 bytes of a slot, which then spans line offsets 40..87 -- TWO lines, two fills, for
// data that fits in one. Aligning the struct to a line makes each slot start at offset 0.
// sizeof is unchanged: 128 is already a multiple of 64.
struct alignas(64) LocalDFBInterface {
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
    uint8_t _tc_align_pad;  // pad bytes [8,20) → 20B so tc_slots[] stays 4B-aligned

    uint16_t num_entries;

    // Hoisted out of DFBTCSlot: six contiguous bytes here cost 6, where one byte per slot cost 18
    // in padding. Also puts all six in one cache line for the TC programming loop.
    dfb::PackedTileCounter ptc[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    uint8_t _pad1[4];  // 28 -> 32 so tc_slots[] is 8B-aligned and the struct reaches 128

    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
};

static_assert(sizeof(DFBTCSlot) == 16, "DFBTCSlot size is incorrect");
// 128 = power of two (array index is a shift, not a multiply) and exactly two 64B cache lines.
// NOTE: the size alone does NOT make a slot line-aligned -- that needs the array base to be
// line-aligned too, which is what the alignas(64) above is for. This comment previously claimed
// slots were line-aligned by virtue of the size; they were not. g_dfb_interface sat at TLS offset
// 0x828 (0x828 % 64 == 40), so every slot started 40 bytes into a line and the 48-byte init write
// straddled two of them.
static_assert(
    alignof(LocalDFBInterface) >= 64,
    "iface must be line-aligned: a 48B slot write at a "
    "non-zero line offset straddles two cache lines");
static_assert(sizeof(LocalDFBInterface) == 128, "LocalDFBInterface size is incorrect");
static_assert(offsetof(LocalDFBInterface, tc_slots) == 32, "tc_slots must start at 32");

// Uniform accessor so call sites do not need to know where the byte lives. On DM it was hoisted
// out of DFBTCSlot (see above); on the TRISC variants it is still inside the slot.
inline dfb::PackedTileCounter dfb_iface_ptc(const LocalDFBInterface& i, uint32_t t) { return i.ptc[t]; }

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
struct TxnDFBDescriptor {
    uint8_t num_counters;
    dfb::PackedTileCounter tile_counters[dfb::MAX_TILE_COUNTERS_PER_SIDE];
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
