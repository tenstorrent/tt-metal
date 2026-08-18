// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared remote-DFB (CrossNode + Persistent) layout constants for host and device.
// Each type has an independent id/slot space of size MAX_REMOTE_DFBS on one Program.

// Sentinel for launch-msg dense-region offsets when no participants of that type are present.
// Valid offsets are L1-aligned and therefore never equal 0xFF.
inline constexpr uint16_t REMOTE_DFB_OFFSET_NONE = 0xFF;

// Max remote DFBs per type per program / per core (ids in [0, MAX)).
inline constexpr uint32_t MAX_REMOTE_DFBS = 16;

// Words per dense kernel-config slot: [config_page_addr, entry_size, relay_dfb_id].
// relay_dfb_id is RELAY_DFB_INVALID when the host did not declare a relay.
inline constexpr uint32_t UINT32_WORDS_PER_REMOTE_DFB_CONFIG = 3;

// Leading word of a dense remote-DFB region: num_slots, then dense slots.
inline constexpr uint32_t REMOTE_DFB_REGION_HEADER_WORDS = 1;

inline constexpr uint32_t remote_dfb_config_region_words(uint32_t num_slots) {
    return REMOTE_DFB_REGION_HEADER_WORDS + num_slots * UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
}

// ---- CrossNode aliases (existing names) ----
inline constexpr uint16_t CROSS_NODE_DFB_OFFSET_NONE = REMOTE_DFB_OFFSET_NONE;
inline constexpr uint32_t MAX_CROSS_NODE_DFBS = MAX_REMOTE_DFBS;
inline constexpr uint32_t UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG = UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
inline constexpr uint32_t CROSS_NODE_DFB_REGION_HEADER_WORDS = REMOTE_DFB_REGION_HEADER_WORDS;
inline constexpr uint32_t CROSS_NODE_DFB_CONFIG_WORDS = UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG;

inline constexpr uint32_t cross_node_dfb_config_region_words(uint32_t num_cross_node_dfbs) {
    return remote_dfb_config_region_words(num_cross_node_dfbs);
}

// ---- Persistent aliases ----
inline constexpr uint16_t PERSISTENT_DFB_OFFSET_NONE = REMOTE_DFB_OFFSET_NONE;
inline constexpr uint32_t MAX_PERSISTENT_DFBS = MAX_REMOTE_DFBS;
inline constexpr uint32_t UINT32_WORDS_PER_PERSISTENT_DFB_CONFIG = UINT32_WORDS_PER_REMOTE_DFB_CONFIG;
inline constexpr uint32_t PERSISTENT_DFB_REGION_HEADER_WORDS = REMOTE_DFB_REGION_HEADER_WORDS;

inline constexpr uint32_t persistent_dfb_config_region_words(uint32_t num_persistent_dfbs) {
    return remote_dfb_config_region_words(num_persistent_dfbs);
}
