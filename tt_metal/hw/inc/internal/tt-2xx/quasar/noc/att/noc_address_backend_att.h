// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file
 * @brief The ATT NoC address backend: implements the noc_address_backend
 * interface over the typed resolution layer and the active map configuration.
 * Every produced value is a complete translated 64-bit operand; nothing here
 * composes or decomposes raw XY coordinates. Include through the arch-selected
 * "noc_address_backend.h" wrapper, which sets the backend-neutral
 * noc_address_backend alias; never include or name this namespace directly.
 *
 * When ATT is enabled it is used for ALL addressing - no mixing. An identity
 * the active map cannot resolve (an out-of-map coordinate, an unbound bank or
 * dispatch key, a local address the window cannot carry) traps unconditionally
 * instead of issuing an operand. System memory has no ATT window on any map
 * and is rejected at compile time.
 */

#include <cstdint>
#include <optional>

#include "api/debug/assert.h"
#include "internal/tt-2xx/quasar/noc/att/att_config.h"

namespace noc_address_backend_att {

/// Unwrap a resolution result. ASSERT names the failure under the watcher;
/// the trap makes it fatal in every build.
FORCE_INLINE uint64_t resolved_or_trap(const std::optional<noc_att::NocAddress>& result) {
    ASSERT(result.has_value());
    if (!result.has_value()) {
        __builtin_trap();
    }
    return *result;
}

/// This initiator's resolved map identity, from the my_x/my_y coordinates
/// firmware latched out of NOC_NODE_ID. Cached after the first lookup: the
/// inverse endpoint search is linear, and is_local sits on hot paths.
FORCE_INLINE noc_att::ResolvedTile current_tile(uint8_t noc) {
    // Cached per NOC to mirror the my_x[noc]/my_y[noc] indexing (Quasar has
    // one NOC today; the array costs nothing and keeps the shape honest).
    // A kernel's copy of this cache may hold what the previous kernel left in
    // the slot: trust it only when it names this tile's latched coordinates.
    static noc_att::ResolvedTile cached[NUM_NOCS] = {};
    if (!cached[noc].valid || cached[noc].noc_x != my_x[noc] || cached[noc].noc_y != my_y[noc]) {
        cached[noc] = noc_att::resolve_current(ACTIVE_ATT_MAP, my_x[noc], my_y[noc]);
    }
    return cached[noc];
}

FORCE_INLINE uint64_t worker_address(uint32_t x, uint32_t y, uint32_t local_address, uint8_t noc) {
    return resolved_or_trap(noc_att::Address::worker(x, y, local_address).encode<ACTIVE_ATT_MAP>());
}

FORCE_INLINE uint64_t self_address(uint32_t local_address, uint8_t noc) {
    // The local window's per-initiator endpoint is boot-patched to this tile,
    // so the operand is one constant OR - no identity lookup.
    return NOC_ATT_LOCAL_WINDOW_BASE | local_address;
}

FORCE_INLINE uint64_t packed_worker_address(uint32_t packed_xy, uint32_t local_address) {
    // The packed word is a host coordinate ((y << NOC_ADDR_NODE_ID_BITS) | x, as
    // in the L1 bank table, go messages and CQ state). resolve_host_coordinate
    // adds the map's frame offset before looking it up in the endpoint tables.
    // DRAM banks do not come through here: bank_address<true> maps a logical
    // bank straight to its selector.
    const uint32_t x = packed_xy & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
    const uint32_t y = (packed_xy >> NOC_ADDR_NODE_ID_BITS) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
    const noc_att::ResolvedTile tile = noc_att::resolve_host_coordinate(ACTIVE_ATT_MAP, x, y);
    ASSERT(tile.valid);
    if (!tile.valid) {
        // WindowClass::Invalid is one past the window array: never index it.
        __builtin_trap();
    }
    const noc_att::Window& window = noc_att::map_window(ACTIVE_ATT_MAP, tile.window);
    ASSERT(window.transfer_supported(local_address));
    if (!window.transfer_supported(local_address)) {
        // A local address past the window's slot would carry into the
        // selector field and reach a different tile.
        __builtin_trap();
    }
    return window.make_address(tile.selector, local_address);
}

/// The packed software multicast descriptor (36-bit local + four 6-bit worker
/// coordinates). A container format only: the V3 issue path decodes it back to
/// worker coordinates and resolves the rectangle through the map.
FORCE_INLINE uint64_t multicast_descriptor(
    uint32_t start_x, uint32_t start_y, uint32_t end_x, uint32_t end_y, uint32_t local_address, uint8_t noc) {
    return noc_att::make_multicast_descriptor(start_x, start_y, end_x, end_y, local_address);
}

template <bool DRAM>
FORCE_INLINE uint64_t bank_address(uint32_t bank_index, uint32_t local_address, uint8_t noc) {
    if constexpr (DRAM) {
        // Typed: logical bank -> the map's DRAM selector. The host's
        // dram_bank_to_noc_xy words are never consulted under ATT.
        return resolved_or_trap(noc_att::Address::dram(bank_index, local_address).encode<ACTIVE_ATT_MAP>());
    } else {
        return packed_worker_address(l1_bank_to_noc_xy[noc][bank_index], local_address);
    }
}

FORCE_INLINE uint32_t extract_local_address(uint64_t address) {
    return static_cast<uint32_t>(resolved_or_trap(noc_att::extract_local_address<ACTIVE_ATT_MAP>(address)));
}

FORCE_INLINE bool is_local(uint64_t address, uint8_t noc) {
    return noc_att::is_self_address<ACTIVE_ATT_MAP>(address, current_tile(noc));
}

/// Whether a kernel-visible (host frame) coordinate names this initiator: the
/// map's frame offset is applied before comparing with the my_x/my_y firmware
/// latched out of NOC_NODE_ID.
FORCE_INLINE bool is_local_coordinate(uint32_t x, uint32_t y, uint8_t noc) {
    return noc_att::host_coordinate_is_current(ACTIVE_ATT_MAP, x, y, my_x[noc], my_y[noc]);
}

// Dispatch go-message coordinates arrive as the raw uint8_t fields of go_msg_t.
FORCE_INLINE uint64_t dispatch_address(uint8_t x, uint8_t y, uint32_t local_address) {
    return resolved_or_trap(noc_att::Address::dispatch(x, y, local_address).encode<ACTIVE_ATT_MAP>());
}

// No ATT map binds a system-memory (PCIe) window; the shared
// get_system_memory_noc_addr wrapper is deleted under NOC_ATT_ENABLED, so this
// interface entry is deleted too rather than silently producing an XY operand.
uint64_t system_memory_address(uint32_t local_address, uint8_t noc) = delete;

}  // namespace noc_address_backend_att
