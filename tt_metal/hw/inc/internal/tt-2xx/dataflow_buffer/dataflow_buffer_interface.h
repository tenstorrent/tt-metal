// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
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
extern thread_local LocalDFBInterface g_dfb_interface[dfb::NUM_DFBS];
#ifndef COMPILE_FOR_TRISC
// TODO: make this a constant when we clean up number of txn ids
extern volatile TxnDFBDescriptor g_txn_dfb_descriptor[32];
extern RemapperAPI g_remapper_configurator;
#endif

// Per–tile-counter slot
struct DFBTCSlot {
    uint32_t rd_ptr;
    uint32_t wr_ptr;
    uint32_t base_addr;
    uint32_t limit;
#ifdef COMPILE_FOR_TRISC
    uint16_t rd_entry_idx;
    uint16_t wr_entry_idx;
    uint16_t base_entry_idx;
#endif
    dfb::PackedTileCounter packed_tile_counter;
} __attribute__((packed));

// on WH/BH arrays will be sized to 1
struct LocalDFBInterface {
    uint32_t entry_size;
    uint32_t stride_size;

#ifdef COMPILE_FOR_TRISC
    uint32_t stride_size_tiles; // used by triscs to calculate tile offset from base L1 address
    uint16_t wr_entry_ptr;
#endif

    uint8_t num_tcs_to_rr;
    uint8_t tc_idx;

#ifdef COMPILE_FOR_TRISC
    uint8_t tensix_trisc_mask;  // which TRISC(s) use this DFB (bit N = trisc N); for runtime gate on TRISC
#else
    uint8_t txn_ids[dfb::NUM_TXN_IDS];
    uint8_t
        num_entries_per_txn_id;
    uint8_t num_entries_per_txn_id_per_tc;
    uint8_t num_txn_ids;
    uint8_t broadcast_tc;       // DM-DM BLOCKED producer: post to all TCs instead of round-robin
#endif

    DFBTCSlot tc_slots[dfb::MAX_NUM_TILE_COUNTERS_TO_RR];
} __attribute__((packed));

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

#ifdef COMPILE_FOR_TRISC
static_assert(sizeof(DFBTCSlot) == 23, "DFBTCSlot size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 109, "LocalDFBInterface size is incorrect");
#else
static_assert(sizeof(DFBTCSlot) == 17, "DFBTCSlot size is incorrect");
static_assert(sizeof(LocalDFBInterface) == 86, "LocalDFBInterface size is incorrect");
#endif
