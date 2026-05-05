// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
#ifndef COMPILE_FOR_TRISC
// TODO: make this a constant when we clean up number of txn ids
extern volatile TxnDFBDescriptor g_txn_dfb_descriptor[32];
extern RemapperAPI g_remapper_configurator;
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
static_assert(sizeof(LocalDFBInterface) == 61, "LocalDFBInterface (pack TRISC) size is incorrect");

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
static_assert(sizeof(LocalDFBInterface) == 60, "LocalDFBInterface (unpack TRISC) size is incorrect");

#else

// Per–tile-counter slot (DM)
struct DFBTCSlot {
    uint32_t rd_ptr;
    uint32_t wr_ptr;
    uint32_t base_addr;
    uint32_t limit;
    dfb::PackedTileCounter packed_tile_counter;
} __attribute__((packed));

// on WH/BH arrays will be sized to 1
struct LocalDFBInterface {
    uint32_t entry_size;
    uint32_t stride_size;

    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;

    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t
        num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t num_txn_ids;
    uint8_t broadcast_tc;  // DM-DM ALL producer: post to all TCs instead of round-robin

    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
} __attribute__((packed));

static_assert(sizeof(DFBTCSlot) == 17, "DFBTCSlot size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 86, "LocalDFBInterface size is incorrect");

#endif

inline LocalDFBInterface& get_local_dfb_interface(uint32_t logical_dfb_id) {
#if defined(COMPILE_FOR_TRISC) && defined(UCK_CHLKC_PACK)
    return g_dfb_interface[g_dfb_logical_to_compact[logical_dfb_id]];
#else
    return g_dfb_interface[logical_dfb_id];
#endif
}

// Holds metadata for transaction ID based ISR handling
// It is used by the ISR to understand which tile counters need to update which credits (post/ack)
struct TxnDFBDescriptor {
    uint8_t num_counters;
    dfb::PackedTileCounter tile_counters[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
    union {
        uint8_t tiles_to_post;
        uint8_t tiles_to_ack;
    } __attribute__((packed));
};
