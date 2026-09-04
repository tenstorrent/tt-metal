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
//   then per slot:  [config_page_addr, entry_size, relay_dfb_id]
//
// Slot ids are uint8_t in [0, 255). Host refuses the next allocate when the
// counter would wrap; 0xFF is also RelayDFBBindingToken::NO_PREFETCHER_PIPE.

// Sentinel for launch-msg dense-region offsets when no participants of that type
// are present. Valid offsets are L1-aligned and therefore never equal 0xFF.
inline constexpr uint16_t REMOTE_DFB_OFFSET_NONE = 0xFF;

// Words per dense kernel-config slot: [config_page_addr, entry_size, relay_dfb_id].
// relay_dfb_id is RELAY_DFB_INVALID when the host did not declare a relay.
inline constexpr uint32_t UINT32_WORDS_PER_REMOTE_DFB_CONFIG = 3;

// Leading word of a dense remote-DFB region: num_slots, then dense slots.
inline constexpr uint32_t REMOTE_DFB_REGION_HEADER_WORDS = 1;

inline constexpr uint32_t remote_dfb_config_region_words(uint32_t num_slots) {
    return REMOTE_DFB_REGION_HEADER_WORDS + num_slots * UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
}
