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
 * When ATT is enabled it is used for ALL addressing - no mixing. Identities
 * the active map does not bind (QSR1 logical DRAM and dispatch until
 * descriptor-owned rows exist) fail resolution and trip the ASSERT; system
 * memory has no ATT window on any map and is rejected at compile time.
 */

#include <cstdint>
#include <optional>

#include "api/debug/assert.h"
#include "internal/tt-2xx/quasar/noc/att/att_config.h"

namespace noc_address_backend_att {

/// This initiator's resolved map identity, from the my_x/my_y coordinates
/// firmware latched out of NOC_NODE_ID. Cached after the first lookup: the
/// inverse endpoint search is linear, and is_local sits on hot paths.
FORCE_INLINE noc_att::ResolvedTile current_tile(uint8_t noc) {
    // Cached per NOC to mirror the my_x[noc]/my_y[noc] indexing (Quasar has
    // one NOC today; the array costs nothing and keeps the shape honest).
    static noc_att::ResolvedTile cached[NUM_NOCS] = {};
    if (!cached[noc].valid) {
        cached[noc] = noc_att::resolve_current(ACTIVE_ATT_MAP, my_x[noc], my_y[noc]);
    }
    return cached[noc];
}

FORCE_INLINE uint64_t worker_address(uint32_t x, uint32_t y, uint32_t local_address, uint8_t noc) {
    const std::optional<noc_att::NocAddress> result =
        noc_att::Address::worker(x, y, local_address).encode<ACTIVE_ATT_MAP>();
    ASSERT(result.has_value());
    return *result;
}

FORCE_INLINE uint64_t self_address(uint32_t local_address, uint8_t noc) {
    // The local window's per-initiator endpoint is boot-patched to this tile,
    // so the operand is one constant OR - no identity lookup.
    return NOC_ATT_LOCAL_WINDOW_BASE | local_address;
}

FORCE_INLINE uint64_t packed_worker_address(uint32_t packed_xy, uint32_t local_address) {
    // Host-generated bank tables pack (y << NOC_ADDR_NODE_ID_BITS) | x in the
    // kernel-visible coordinate frame - and they carry BOTH worker (L1 bank)
    // and DRAM tile coordinates through this one entry point. A packed
    // coordinate is exactly an endpoint word, so resolve it the way
    // resolve_current does: the worker table first, then the full-tile table
    // (which covers the DRAM/perimeter tiles). On the aether map a DRAM tile
    // resolves to the same remote-window selector Address::dram produces.
    const uint32_t x = packed_xy & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
    const uint32_t y = (packed_xy >> NOC_ADDR_NODE_ID_BITS) & ((1u << NOC_ADDR_NODE_ID_BITS) - 1);
    const noc_att::ResolvedTile tile = noc_att::resolve_current(ACTIVE_ATT_MAP, x, y);
    ASSERT(tile.valid);
    const noc_att::Window& window = noc_att::map_window(ACTIVE_ATT_MAP, tile.window);
    ASSERT(window.transfer_supported(local_address));
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
        const std::optional<noc_att::NocAddress> result =
            noc_att::Address::dram(bank_index, local_address).encode<ACTIVE_ATT_MAP>();
        ASSERT(result.has_value());
        return *result;
    } else {
        return packed_worker_address(l1_bank_to_noc_xy[noc][bank_index], local_address);
    }
}

FORCE_INLINE uint32_t extract_local_address(uint64_t address) {
    const std::optional<noc_att::NocAddress> result = noc_att::extract_local_address<ACTIVE_ATT_MAP>(address);
    ASSERT(result.has_value());
    return static_cast<uint32_t>(*result);
}

FORCE_INLINE bool is_local(uint64_t address, uint8_t noc) {
    return noc_att::is_self_address<ACTIVE_ATT_MAP>(address, current_tile(noc));
}

// Dispatch go-message coordinates arrive as the raw uint8_t fields of go_msg_t.
FORCE_INLINE uint64_t dispatch_address(uint8_t x, uint8_t y, uint32_t local_address) {
    const std::optional<noc_att::NocAddress> result =
        noc_att::Address::dispatch(x, y, local_address).encode<ACTIVE_ATT_MAP>();
    ASSERT(result.has_value());
    return *result;
}

// No ATT map binds a system-memory (PCIe) window; the shared
// get_system_memory_noc_addr wrapper is deleted under NOC_ATT_ENABLED, so this
// interface entry is deleted too rather than silently producing an XY operand.
uint64_t system_memory_address(uint32_t local_address, uint8_t noc) = delete;

}  // namespace noc_address_backend_att
