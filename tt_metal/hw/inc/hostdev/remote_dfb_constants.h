// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Launch-msg dense index for CrossNodeDFB and PrefetcherPipe (host and device).
// This is not the per-core L1 config page; see remote_dfb_config_layout.h.
//
// kernel_config_msg_t has one dense region per type (cross_node_dfb_offset,
// prefetcher_pipe_offset). Both regions use this layout:
//   word[0]         num_slots
//   then per slot:  [config_page_addr, entry_size, relay_word]
//   relay_word:     bits  7:0  relay_dfb_id (RELAY_DFB_INVALID when no relay)
//                   bits 23:16 PrefetcherPipe active credit lanes P; 0 means 1
//                              (CrossNode leaves these 0)
//
// Slot ids are uint8_t in [0, 255). Host refuses the next allocate when the
// counter would wrap; 0xFF is also RelayDFBBindingToken::NO_PREFETCHER_PIPE.
//
// P rides in the per-program slot rather than in the persistent config page so it is
// delivered in command-queue order with the program that uses it: a host poke into
// persistent L1 is not ordered against programs already in flight on the same pipe.

// Sentinel for launch-msg dense-region offsets when no participants of that type
// are present. Valid offsets are L1-aligned and therefore never equal 0xFF.
inline constexpr uint16_t REMOTE_DFB_OFFSET_NONE = 0xFF;

// Words per dense kernel-config slot: [config_page_addr, entry_size, relay_word].
inline constexpr uint32_t UINT32_WORDS_PER_REMOTE_DFB_CONFIG = 3;

inline constexpr uint32_t REMOTE_DFB_SLOT_RELAY_ID_MASK = 0xFFu;
inline constexpr uint32_t PREFETCHER_PIPE_SLOT_CREDIT_LANES_SHIFT = 16;
inline constexpr uint32_t PREFETCHER_PIPE_SLOT_CREDIT_LANES_MASK = 0xFFu;

inline constexpr uint32_t pack_prefetcher_pipe_slot_relay_word(uint32_t relay_dfb_id, uint32_t num_credit_lanes) {
    // P == 1 encodes as 0 so a single-lane slot is bit-identical to one without the field.
    return (relay_dfb_id & REMOTE_DFB_SLOT_RELAY_ID_MASK) |
           ((num_credit_lanes > 1 ? num_credit_lanes & PREFETCHER_PIPE_SLOT_CREDIT_LANES_MASK : 0u)
            << PREFETCHER_PIPE_SLOT_CREDIT_LANES_SHIFT);
}

inline constexpr uint32_t prefetcher_pipe_slot_relay_id(uint32_t relay_word) {
    return relay_word & REMOTE_DFB_SLOT_RELAY_ID_MASK;
}

inline constexpr uint32_t prefetcher_pipe_slot_credit_lanes(uint32_t relay_word) {
    const uint32_t lanes =
        (relay_word >> PREFETCHER_PIPE_SLOT_CREDIT_LANES_SHIFT) & PREFETCHER_PIPE_SLOT_CREDIT_LANES_MASK;
    return lanes == 0 ? 1u : lanes;
}

// Leading word of a dense remote-DFB region: num_slots, then dense slots.
inline constexpr uint32_t REMOTE_DFB_REGION_HEADER_WORDS = 1;

inline constexpr uint32_t remote_dfb_config_region_words(uint32_t num_slots) {
    return REMOTE_DFB_REGION_HEADER_WORDS + num_slots * UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
}
