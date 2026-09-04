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
// Slot ids are uint8_t in [0, 254): the top two values are the reserved
// PREFETCHER_PIPE_ID_* sentinels below, and the host refuses to allocate a slot
// that would reach them.

// Reserved prefetcher-pipe ids a relay DFB binding may carry instead of a slot id.
// Both are RelayDFBBindingToken values baked into a kernel at JIT time.
//  * NONE: no pipe at all (a CrossNode relay). The kernel must not align to a
//    durable checkpoint -- CrossNode state is re-zeroed every launch.
//  * BY_RELAY: exactly one pipe on this core relays through this DFB, but which
//    one differs per core (one relay DFB spanning the receivers of several
//    pipes). The kernel finds its slot by scanning the dense region for the one
//    whose relay_dfb_id is this DFB's, which is a per-core fact the binary cannot
//    bake in -- and must not, since the JIT cache key does not fold the pipe id.
inline constexpr uint8_t PREFETCHER_PIPE_ID_NONE = 0xFF;
inline constexpr uint8_t PREFETCHER_PIPE_ID_BY_RELAY = 0xFE;

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
