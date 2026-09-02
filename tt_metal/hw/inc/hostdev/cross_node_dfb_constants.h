// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Shared CrossNodeDFB layout constants for host and device.

// Sentinel for kernel_config_msg_t::cross_node_dfb_offset when no CrossNodeDFB participants
// are present on the kernel group.
// Valid offsets are L1-aligned and therefore never equal 0xFF.
inline constexpr uint16_t CROSS_NODE_DFB_OFFSET_NONE = 0xFF;

// Max CrossNodeDFBs per program / per core (remote_dfb_id in [0, MAX)).
inline constexpr uint32_t MAX_CROSS_NODE_DFBS = 16;

// Words per dense kernel-config slot: [config_page_addr, entry_size, relay_dfb_id].
// relay_dfb_id is RELAY_DFB_INVALID when the host did not declare a relay.
inline constexpr uint32_t UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG = 3;

// Leading word of the kernel-config CrossNode region: num_slots, then dense slots.
inline constexpr uint32_t CROSS_NODE_DFB_REGION_HEADER_WORDS = 1;

inline constexpr uint32_t CROSS_NODE_DFB_CONFIG_WORDS = UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG;

inline constexpr uint32_t cross_node_dfb_config_region_words(uint32_t num_cross_node_dfbs) {
    return CROSS_NODE_DFB_REGION_HEADER_WORDS + num_cross_node_dfbs * UINT32_WORDS_PER_CROSS_NODE_DFB_CONFIG;
}
