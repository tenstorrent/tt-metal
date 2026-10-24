// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include "dataflow_buffer_config.h"
#include "core_config.h"  // MaxDMProcessorsPerCoreType
#ifndef COMPILE_FOR_TRISC
#include "internal/tt-2xx/quasar/overlay/remapper_api.hpp"
#endif

// Forward declarations
struct LocalDFBInterface;
struct TxnDFBDescriptor;

// Global DFB interface array. Compute TRISCs keep only the active DFBs, in a compact array reached
// through g_dfb_logical_to_compact; DM harts index by logical id.
#if defined(COMPILE_FOR_TRISC)
extern thread_local LocalDFBInterface g_dfb_interface[dfb::MAX_ACTIVE_DFBS_TRISC];
extern thread_local uint8_t g_dfb_logical_to_compact[dfb::NUM_DFBS];
#else
extern thread_local LocalDFBInterface g_dfb_interface[dfb::NUM_DFBS];
#endif

// Cached L1 byte address of the DFB config region; set once during setup_local_dfb_interfaces
// so DataflowBuffer::DataflowBuffer() can call dfb_ensure_ready without a separate parameter.
extern thread_local uintptr_t g_dfb_config_base_addr;
// Logical ids of the DFBs this hart is bound to this launch (bit per id); set by
// setup_local_dfb_interfaces, read by the kernel-exit finish.
extern thread_local uint32_t g_dfb_bound_mask;
#ifndef COMPILE_FOR_TRISC
extern volatile TxnDFBDescriptor g_txn_dfb_descriptor[32];
extern overlay::RemapperAPI g_remapper_configurator;
#endif

#if defined(COMPILE_FOR_TRISC)
// Compute-thread slot (pack and unpack). ring_size is uint32 L1-aligned units so a DFB can span
// full Quasar L1 (~4 MB). cursor_entry_idx is the write cursor on pack and the read cursor on
// unpack; its byte offset is not stored but derived (advances in lockstep with stride_size /
// stride_size_tiles).
struct DFBTCSlot {
    uint32_t base_addr;
    uint32_t ring_size;
    uint16_t cursor_entry_idx;
    uint16_t base_entry_idx;
    dfb::PackedTileCounter packed_tile_counter;
} __attribute__((packed));
#else
struct DFBTCSlot {
    uint32_t ptr;
    uint32_t base_addr;
    uint32_t limit;
    dfb::PackedTileCounter packed_tile_counter;
    uint8_t _align[3];  // pad 13 → 16B so every slot in tc_slots[] is 4B-aligned
};

#endif

#if defined(COMPILE_FOR_TRISC)
static_assert(sizeof(DFBTCSlot) == 13, "DFBTCSlot (TRISC) size is incorrect");
// Compute TRISCs draw their tile-counter slots from one shared pool per thread, so an interface entry
// holds only the index of its contiguous slice (slot_base) and keeps the iface.tc_slots[i] API.
extern thread_local DFBTCSlot g_dfb_tc_slots[dfb::MAX_TRISC_TC_SLOTS];
struct DFBTCSlotSpan {
    uint8_t slot_base;
    inline __attribute__((always_inline)) DFBTCSlot& operator[](uint32_t index) {
        return g_dfb_tc_slots[slot_base + index];
    }
    inline __attribute__((always_inline)) const DFBTCSlot& operator[](uint32_t index) const {
        return g_dfb_tc_slots[slot_base + index];
    }
} __attribute__((packed));
static_assert(sizeof(DFBTCSlotSpan) == 1, "DFBTCSlotSpan size is incorrect");
#endif

#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
struct LocalDFBInterface {
    uint16_t entry_size;
    uint16_t stride_size;
    uint16_t num_entries;
    uint16_t wr_entry_ptr;
    uint8_t stride_size_tiles;
    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;
    uint16_t block_size;  // entries in one block; 1 unless the ring is BLOCKED
    uint16_t split_tc;    // one op (a whole block) is split across all of this hart's TCs
    uint16_t jump;        // cursor jump in entries, taken on each TC rotation
    DFBTCSlotSpan tc_slots;
} __attribute__((packed));
static_assert(sizeof(LocalDFBInterface) == 18, "LocalDFBInterface (pack TRISC) size is incorrect");
#elif defined(COMPILE_FOR_TRISC)
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
    DFBTCSlotSpan tc_slots;
} __attribute__((packed));
static_assert(sizeof(LocalDFBInterface) == 17, "LocalDFBInterface (unpack TRISC) size is incorrect");
#else
struct LocalDFBInterface {
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

namespace dfb {
// DM0 (ISR) and DM1 (remapper) never run kernels, so kernel DMs are DM2 onward.
constexpr uint8_t FIRST_KERNEL_DM_HART = 2;
constexpr uint8_t NUM_KERNEL_DM_HARTS = MaxDMProcessorsPerCoreType - FIRST_KERNEL_DM_HART;
static_assert(NUM_KERNEL_DM_HARTS == MAX_PRODUCERS_PER_DFB, "kernel DMs and DFB producer slots must agree");
}  // namespace dfb

// Implicit-sync progress of this DM on one DFB. A DM hart is either a producer or a consumer of a given DFB,
// never both (DM self-loops are rejected), so one struct covers whichever side it is.
// Firmware hands it to DM0 when the kernel returns and zeroes it for the next launch (dfb_dm_kernel_exit).
struct DFBImplicitProgress {
    uint16_t loop;      // completed txn-id batches, mod 2^16 like the tile counters
    uint8_t pos;        // entries issued into the current batch, < num_entries_per_txn_id
    uint8_t share;      // entries per op; DM0 needs it to split the last batch over the counters
    uint8_t txn_idx;    // index into txn_ids[] of the current batch
    uint8_t share_pos;  // STRIDED side: entries of the current op issued so far
    uint8_t _pad[2];
};
static_assert(sizeof(DFBImplicitProgress) == 8, "DFBImplicitProgress size is incorrect");
extern thread_local DFBImplicitProgress g_dfb_progress[dfb::NUM_DFBS];

// Kernel-exit handoff from each kernel DM to DM0.
// A record holds a DM's loop, pos and share on one implicit-sync DFB side, indexed by the
// side's lowest txn id. A DM signals that its records are written through its run state
// (RUN_SYNC_MSG_DFB_FINISHING).
extern uint32_t g_dfb_exit_record[dfb::NUM_KERNEL_DM_HARTS][NUM_DFB_POOL_TXN_IDS];

#endif

#if defined(COMPILE_FOR_TRISC)
// Byte-offset (L1 units) of the current FIFO cursor from slot.base_addr.
// cursor_entry_idx and the former byte offset advance in lockstep:
//   entry += n * stride_size_tiles;  offset += n * stride_size;
// so offset is recoverable without storing it (saves 4B/slot in compute-TRISC TLS).
inline __attribute__((always_inline)) uint32_t
dfb_slot_cursor_offset_units(const LocalDFBInterface& intf, const DFBTCSlot& slot, uint32_t entry_idx) {
    const uint32_t delta_entries = entry_idx - static_cast<uint32_t>(slot.base_entry_idx);
    return (delta_entries / static_cast<uint32_t>(intf.stride_size_tiles)) * static_cast<uint32_t>(intf.stride_size);
}
#endif

inline LocalDFBInterface& get_local_dfb_interface(uint32_t logical_dfb_id) {
#if defined(COMPILE_FOR_TRISC)
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
    dfb_dm0_finish_info_t finish;  // read only by DM0's kernel-exit flush, not by the ISR
    uint8_t _pad[10];              // pad 22 → 32 bytes
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
static_assert(
    offsetof(TxnDFBDescriptor, finish) == offsetof(dfb_dm0_txn_descriptor_image_t, finish),
    "TxnDFBDescriptor finish must match dfb_dm0_txn_descriptor_image_t finish");
