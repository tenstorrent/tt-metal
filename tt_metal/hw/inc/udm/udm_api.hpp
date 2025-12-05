// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include "tt_metal/hw/inc/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#include <type_traits>
#include "debug/dprint.h"

namespace tt::tt_fabric::experimental::udm {

/**
 * @brief Helper to check if destination fabric node is local
 *
 * Caches the local chip's mesh_id and chip_id from routing table
 */
inline bool dest_is_local(uint32_t dest_fabric_mesh_id, uint32_t dest_fabric_chip_id) {
    static bool initialized = false;
    static uint32_t my_fabric_mesh_id = 0;
    static uint32_t my_fabric_chip_id = 0;

    if (!initialized) {
        auto* routing_table = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
        my_fabric_chip_id = routing_table->my_device_id;
        my_fabric_mesh_id = routing_table->my_mesh_id;
        initialized = true;
    }

    return (my_fabric_mesh_id == dest_fabric_mesh_id) && (my_fabric_chip_id == dest_fabric_chip_id);
}

// ==================== Unified Mesh Accessor APIs ====================

/**
 * @brief Async read from mesh accessor (handles both local and remote)
 *
 * Works with both MeshTensorAccessor (using page_id) and MeshGcoreAccessor (using coord).
 *
 * Usage examples:
 *   async_read(accessor, page_id, l1_addr, size);              // simple read
 *   async_read(accessor, gcore_coord, l1_addr, size, offset);  // with offset
 *
 * @tparam AccessorT MeshTensorAccessor or MeshGcoreAccessor type (auto-deduced)
 * @tparam CoordT Coordinate type (auto-deduced: uint32_t for page_id, array-like for gcore_coord)
 * @param accessor Accessor instance
 * @param coord Global coordinate (page_id or gcore_coord)
 * @param src_addr Local L1 address to receive the data
 * @param size Size of data to read
 * @param offset Offset within the page/core (default 0)
 * @param noc NOC index to use
 */
template <typename AccessorT, typename CoordT>
inline void async_read(
    const AccessorT& accessor,
    const CoordT& coord,
    uint32_t src_addr,
    uint32_t size,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);

    if (dest_is_local(fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id)) {
        // Local transaction - use NOC directly
        noc_async_read(fabric_noc_info.noc_addr, src_addr, size);
    } else {
        // Remote transaction - use fabric
        tt::tt_fabric::udm::fabric_fast_read_any_len(
            fabric_noc_info.fabric_chip_id, fabric_noc_info.fabric_mesh_id, fabric_noc_info.noc_addr, src_addr, size);
    }
}

/**
 * @brief Async write to mesh accessor (handles both local and remote)
 *
 * Works with both MeshTensorAccessor (using page_id) and MeshGcoreAccessor (using coord).
 *
 * Usage examples:
 *   async_write(accessor, page_id, l1_addr, size);              // non-posted (default)
 *   async_write<1>(accessor, page_id, l1_addr, size);           // posted write
 *   async_write(accessor, gcore_coord, l1_addr, size, offset);  // with offset
 *
 * @tparam posted Posted write flag (0 = non-posted, 1 = posted, default 0)
 * @tparam AccessorT MeshTensorAccessor or MeshGcoreAccessor type (auto-deduced)
 * @tparam CoordT Coordinate type (auto-deduced: uint32_t for page_id, array-like for gcore_coord)
 * @param accessor Accessor instance
 * @param coord Global coordinate (page_id or gcore_coord)
 * @param src_addr Local L1 address to send data
 * @param size Size of data to write
 * @param offset Offset within the page/core (default 0)
 * @param noc NOC index to use
 */
template <uint8_t posted = 0, typename AccessorT, typename CoordT>
inline void async_write(
    const AccessorT& accessor,
    const CoordT& coord,
    uint32_t src_addr,
    uint32_t size,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);

    if (dest_is_local(fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id)) {
        // Local transaction - use NOC directly
        noc_async_write(src_addr, fabric_noc_info.noc_addr, size);
    } else {
        // Remote transaction - use fabric
        tt::tt_fabric::udm::fabric_fast_write_any_len(
            fabric_noc_info.fabric_chip_id,
            fabric_noc_info.fabric_mesh_id,
            src_addr,
            fabric_noc_info.noc_addr,
            size,
            false,  // multicast
            1,      // num_dests
            0,      // trid
            posted);
    }
}

// ==================== Barrier APIs ====================

/**
 * @brief Async read barrier (handles both local and remote)
 *
 * Waits for all outstanding read operations to complete, both local NOC reads
 * and remote fabric reads. Always calls both barriers to ensure complete synchronization.
 *
 * @param noc NOC index to use for local barrier
 */
inline void async_read_barrier(uint8_t noc = noc_index) {
    noc_async_read_barrier(noc);
    tt::tt_fabric::udm::fabric_read_barrier();
}

/**
 * @brief Async write barrier (handles both local and remote)
 *
 * Waits for all outstanding write operations to complete, both local NOC writes
 * and remote fabric writes. Always calls both barriers to ensure complete synchronization.
 *
 * @param noc NOC index to use for local barrier
 */
inline void async_write_barrier(uint8_t noc = noc_index) {
    noc_async_write_barrier(noc);
    tt::tt_fabric::udm::fabric_write_barrier();
}

}  // namespace tt::tt_fabric::experimental::udm
