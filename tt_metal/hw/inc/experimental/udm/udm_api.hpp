// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/udm/tt_fabric_udm.hpp"
#include "experimental/udm/accessor/mesh_gcore_accessor.h"
#include "experimental/udm/accessor/mesh_tensor_accessor.h"
#include <type_traits>
#include "api/debug/dprint.h"

namespace tt::tt_fabric::experimental::udm {

// Type trait to detect MeshTensorAccessor (used to exclude it from coord-only overloads)
template <typename T>
struct is_mesh_tensor_accessor : std::false_type {};

template <typename TensorAccessorT, uint32_t Rank, uint32_t NumGrids>
struct is_mesh_tensor_accessor<MeshTensorAccessor<TensorAccessorT, Rank, NumGrids>> : std::true_type {};

/**
 * @brief Helper to check if destination fabric node is local
 *
 * Caches the local chip's mesh_id and chip_id from routing table
 */
FORCE_INLINE bool dest_is_local(uint32_t dest_fabric_mesh_id, uint32_t dest_fabric_chip_id) {
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

// ==================== Implementation Functions ====================

/**
 * @brief Implementation of async read (handles both local and remote)
 *
 * @param fabric_mesh_id Fabric mesh ID of destination
 * @param fabric_chip_id Fabric chip ID of destination
 * @param noc_addr NOC address of destination
 * @param local_addr Local L1 address to receive the data
 * @param size Size of data to read
 */
FORCE_INLINE void async_read_impl(
    uint32_t fabric_mesh_id, uint32_t fabric_chip_id, uint64_t noc_addr, uint32_t local_addr, uint32_t size) {
    if (dest_is_local(fabric_mesh_id, fabric_chip_id)) {
        // Local transaction - use NOC directly
        noc_async_read(noc_addr, local_addr, size);
    } else {
        // Remote transaction - use fabric
        tt::tt_fabric::udm::fabric_fast_read_any_len(fabric_chip_id, fabric_mesh_id, noc_addr, local_addr, size);
    }
}

/**
 * @brief Implementation of async write (handles both local and remote)
 *
 * @tparam posted Posted write flag (0 = non-posted, 1 = posted)
 * @param fabric_mesh_id Fabric mesh ID of destination
 * @param fabric_chip_id Fabric chip ID of destination
 * @param noc_addr NOC address of destination
 * @param src_addr Local L1 address to send data from
 * @param size Size of data to write
 */
template <uint8_t posted = 0>
FORCE_INLINE void async_write_impl(
    uint32_t fabric_mesh_id, uint32_t fabric_chip_id, uint64_t noc_addr, uint32_t src_addr, uint32_t size) {
    if (dest_is_local(fabric_mesh_id, fabric_chip_id)) {
        // Local transaction - use NOC directly
        noc_async_write(src_addr, noc_addr, size);
    } else {
        // Remote transaction - use fabric
        tt::tt_fabric::udm::fabric_fast_write_any_len(
            fabric_chip_id,
            fabric_mesh_id,
            src_addr,
            noc_addr,
            size,
            false,  // multicast
            1,      // num_dests
            0,      // trid
            posted);
    }
}

/**
 * @brief Implementation of semaphore increment (handles both local and remote)
 *
 * @tparam posted Posted atomic flag (0 = non-posted, 1 = posted)
 * @param fabric_mesh_id Fabric mesh ID of destination
 * @param fabric_chip_id Fabric chip ID of destination
 * @param noc_addr NOC address of destination semaphore
 * @param incr_val Value to increment the semaphore by
 * @param noc NOC index to use
 */
template <uint8_t posted = 0>
FORCE_INLINE void semaphore_inc_impl(
    uint32_t fabric_mesh_id, uint32_t fabric_chip_id, uint64_t noc_addr, uint32_t incr_val, uint8_t noc) {
    if (dest_is_local(fabric_mesh_id, fabric_chip_id)) {
        // Local transaction - use NOC atomic increment
        noc_semaphore_inc<posted>(noc_addr, incr_val, noc);
    } else {
        // Remote transaction - use fabric atomic increment
        tt::tt_fabric::udm::fabric_fast_atomic_inc(
            fabric_chip_id,
            fabric_mesh_id,
            incr_val,
            noc_addr,
            0,  // trid
            posted);
    }
}

// ==================== GlobalCore Accessor Management ====================

/**
 * @brief Get or initialize the global gcore accessor (lazy initialization)
 *
 * The gcore accessor is stored as a static variable and initialized once on first use.
 * This follows the same pattern as get_or_open_fabric_connection().
 *
 * The accessor is constructed from compile-time defines (MESH_NUM_DIMS, GRID_NUM_DIMS, etc.)
 * which are already set up by the time the kernel runs.
 *
 * @return Pair of [accessor reference, is_newly_initialized]
 */
FORCE_INLINE std::pair<DefaultMeshGlobalCoreAccessor&, bool> get_or_init_gcore_accessor() {
    static DefaultMeshGlobalCoreAccessor* accessor_ptr = nullptr;
    static bool initialized = false;

    if (!initialized) {
        static DefaultMeshGlobalCoreAccessor accessor;
        auto gcore_args = MeshGlobalCoreAccessorArgs();
        accessor.init(gcore_args);
        accessor_ptr = &accessor;
        initialized = true;
    }

    return {*accessor_ptr, initialized};
}

// ==================== Unified Mesh Accessor APIs ====================

/**
 * @brief Async read from MeshTensorAccessor (handles both local and remote)
 *
 * Usage examples:
 *   async_read(accessor, page_id, l1_addr, size);              // simple read
 *   async_read(accessor, page_id, l1_addr, size, offset);      // with offset
 *
 * @tparam TensorAccessorT Underlying TensorAccessor type (auto-deduced)
 * @tparam Rank Compile-time rank of the tensor (auto-deduced)
 * @tparam NumGrids Compile-time number of grids (auto-deduced)
 * @tparam CoordT Coordinate type (auto-deduced: uint32_t for page_id)
 * @param accessor MeshTensorAccessor instance
 * @param coord Global page_id
 * @param src_addr Local L1 address to receive the data
 * @param size Size of data to read
 * @param offset Offset within the page (default 0)
 * @param noc NOC index to use
 */
template <typename TensorAccessorT, uint32_t Rank, uint32_t NumGrids, typename CoordT>
FORCE_INLINE void async_read(
    const MeshTensorAccessor<TensorAccessorT, Rank, NumGrids>& accessor,
    const CoordT& coord,
    uint32_t src_addr,
    uint32_t size,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);
    async_read_impl(
        fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id, fabric_noc_info.noc_addr, src_addr, size);
}

/**
 * @brief Async write to MeshTensorAccessor (handles both local and remote)
 *
 * Usage examples:
 *   async_write(accessor, page_id, l1_addr, size);              // non-posted (default)
 *   async_write<1>(accessor, page_id, l1_addr, size);           // posted write
 *   async_write(accessor, page_id, l1_addr, size, offset);      // with offset
 *
 * @tparam posted Posted write flag (0 = non-posted, 1 = posted, default 0)
 * @tparam TensorAccessorT Underlying TensorAccessor type (auto-deduced)
 * @tparam Rank Compile-time rank of the tensor (auto-deduced)
 * @tparam NumGrids Compile-time number of grids (auto-deduced)
 * @tparam CoordT Coordinate type (auto-deduced: uint32_t for page_id)
 * @param accessor MeshTensorAccessor instance
 * @param coord Global page_id
 * @param src_addr Local L1 address to send data
 * @param size Size of data to write
 * @param offset Offset within the page (default 0)
 * @param noc NOC index to use
 */
template <uint8_t posted = 0, typename TensorAccessorT, uint32_t Rank, uint32_t NumGrids, typename CoordT>
FORCE_INLINE void async_write(
    const MeshTensorAccessor<TensorAccessorT, Rank, NumGrids>& accessor,
    const CoordT& coord,
    uint32_t src_addr,
    uint32_t size,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);
    async_write_impl<posted>(
        fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id, fabric_noc_info.noc_addr, src_addr, size);
}

/**
 * @brief Async read from gcore using global accessor (handles both local and remote)
 *
 * Uses the global gcore accessor (lazy-initialized on first call).
 * For local gcores, uses NOC. For remote gcores (across devices), uses fabric.
 *
 * This overload is for gcore-to-gcore reads where coord is an array-like gcore coordinate.
 * The accessor parameter version is for explicit accessor use with tensors.
 *
 * Usage example:
 *   async_read(source_gcore_coord, l1_addr, size, offset);
 *
 * @tparam CoordT Coordinate type (array-like for gcore_coord, not uint32_t)
 * @param coord Global gcore coordinate (must be array-like, not page_id)
 * @param src_addr Local L1 address to receive the data
 * @param size Size of data to read
 * @param offset Offset within the gcore's L1 (default 0)
 * @param noc NOC index to use
 */
template <
    typename CoordT,
    typename = std::enable_if_t<!std::is_same_v<CoordT, uint32_t> && !is_mesh_tensor_accessor<CoordT>::value>>
FORCE_INLINE void async_read(
    const CoordT& coord, uint32_t src_addr, uint32_t size, uint32_t offset = 0, uint8_t noc = noc_index) {
    auto [accessor, is_init] = get_or_init_gcore_accessor();
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);
    async_read_impl(
        fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id, fabric_noc_info.noc_addr, src_addr, size);
}

/**
 * @brief Async write to gcore using global accessor (handles both local and remote)
 *
 * Uses the global gcore accessor (lazy-initialized on first call).
 * For local gcores, uses NOC. For remote gcores (across devices), uses fabric.
 *
 * This overload is for gcore-to-gcore writes where coord is an array-like gcore coordinate.
 * The accessor parameter version is for explicit accessor use with tensors.
 *
 * Usage example:
 *   async_write(target_gcore_coord, l1_addr, size, offset);
 *
 * @tparam posted Posted write flag (0 = non-posted, 1 = posted, default 0)
 * @tparam CoordT Coordinate type (array-like for gcore_coord, not uint32_t)
 * @param coord Global gcore coordinate (must be array-like, not page_id)
 * @param src_addr Local L1 address to send data
 * @param size Size of data to write
 * @param offset Offset within the gcore's L1 (default 0)
 * @param noc NOC index to use
 */
template <
    uint8_t posted = 0,
    typename CoordT,
    typename = std::enable_if_t<!std::is_same_v<CoordT, uint32_t> && !is_mesh_tensor_accessor<CoordT>::value>>
FORCE_INLINE void async_write(
    const CoordT& coord, uint32_t src_addr, uint32_t size, uint32_t offset = 0, uint8_t noc = noc_index) {
    auto [accessor, is_init] = get_or_init_gcore_accessor();
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);
    async_write_impl<posted>(
        fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id, fabric_noc_info.noc_addr, src_addr, size);
}

// ==================== Semaphore APIs ====================

/**
 * @brief Atomic semaphore increment using MeshTensorAccessor (handles both local and remote)
 *
 * For local pages, uses NOC atomic increment. For remote pages (across devices),
 * uses fabric atomic increment.
 *
 * Usage example:
 *   semaphore_inc(accessor, page_id, 1, semaphore_offset);
 *
 * @tparam posted Posted atomic flag (0 = non-posted, 1 = posted, default 0)
 * @tparam TensorAccessorT Underlying TensorAccessor type (auto-deduced)
 * @tparam Rank Compile-time rank of the tensor (auto-deduced)
 * @tparam NumGrids Compile-time number of grids (auto-deduced)
 * @tparam CoordT Coordinate type (auto-deduced: uint32_t for page_id)
 * @param accessor MeshTensorAccessor instance
 * @param coord Global page_id
 * @param incr_val Value to increment the semaphore by
 * @param offset Offset within the page (semaphore address, default 0)
 * @param noc NOC index to use
 */
template <uint8_t posted = 0, typename TensorAccessorT, uint32_t Rank, uint32_t NumGrids, typename CoordT>
FORCE_INLINE void semaphore_inc(
    const MeshTensorAccessor<TensorAccessorT, Rank, NumGrids>& accessor,
    const CoordT& coord,
    uint32_t incr_val,
    uint32_t offset = 0,
    uint8_t noc = noc_index) {
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);
    semaphore_inc_impl<posted>(
        fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id, fabric_noc_info.noc_addr, incr_val, noc);
}

/**
 * @brief Atomic semaphore increment using global accessor (handles both local and remote)
 *
 * Uses the global gcore accessor (lazy-initialized on first call).
 * For local gcores, uses NOC atomic increment. For remote gcores, uses fabric atomic increment.
 *
 * This overload is for gcore-to-gcore semaphore operations where coord is an array-like gcore coordinate.
 * The accessor parameter version is for explicit accessor use.
 *
 * Usage example:
 *   semaphore_inc(target_gcore_coord, 1, semaphore_offset);
 *
 * @tparam posted Posted atomic flag (0 = non-posted, 1 = posted, default 0)
 * @tparam CoordT Coordinate type (array-like for gcore_coord, not uint32_t or AccessorT)
 * @param coord Global gcore coordinate (must be array-like, not an accessor)
 * @param incr_val Value to increment the semaphore by
 * @param offset Offset within the gcore's L1 (semaphore address, default 0)
 * @param noc NOC index to use
 */
template <
    uint8_t posted = 0,
    typename CoordT,
    typename = std::enable_if_t<!std::is_same_v<CoordT, uint32_t> && !is_mesh_tensor_accessor<CoordT>::value>>
FORCE_INLINE void semaphore_inc(const CoordT& coord, uint32_t incr_val, uint32_t offset = 0, uint8_t noc = noc_index) {
    auto [accessor, is_init] = get_or_init_gcore_accessor();
    auto fabric_noc_info = accessor.get_fabric_node_and_noc_addr(coord, offset, noc);
    semaphore_inc_impl<posted>(
        fabric_noc_info.fabric_mesh_id, fabric_noc_info.fabric_chip_id, fabric_noc_info.noc_addr, incr_val, noc);
}

/**
 * @brief Wait for semaphore to reach target value
 *
 * Wrapper around noc_semaphore_wait for consistency with UDM API.
 * This is a local operation on the current core's semaphore.
 *
 * Usage example:
 *   semaphore_wait(semaphore_addr, expected_count);
 *
 * @param semaphore_addr L1 address of the semaphore
 * @param target_value Value to wait for
 */
FORCE_INLINE void semaphore_wait(uint32_t semaphore_addr, uint32_t target_value) {
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr), target_value);
}

/**
 * @brief Set semaphore to a specific value
 *
 * Wrapper around noc_semaphore_set for consistency with UDM API.
 * This is a local operation on the current core's semaphore.
 *
 * Usage example:
 *   semaphore_set(semaphore_addr, 0);  // Reset semaphore
 *
 * @param semaphore_addr L1 address of the semaphore
 * @param value Value to set the semaphore to
 */
FORCE_INLINE void semaphore_set(uint32_t semaphore_addr, uint32_t value) {
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr), value);
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
FORCE_INLINE void async_read_barrier(uint8_t noc = noc_index) {
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
FORCE_INLINE void async_write_barrier(uint8_t noc = noc_index) {
    noc_async_write_barrier(noc);
    tt::tt_fabric::udm::fabric_write_barrier();
}

/**
 * @brief Atomic operation barrier (handles both local and remote)
 *
 * Waits for all outstanding atomic operations to complete, both local NOC atomics
 * and remote fabric atomics. Always calls both barriers to ensure complete synchronization.
 *
 * @param noc NOC index to use for local barrier
 */
FORCE_INLINE void atomic_barrier(uint8_t noc = noc_index) {
    noc_async_atomic_barrier(noc);
    tt::tt_fabric::udm::fabric_atomic_barrier();
}

}  // namespace tt::tt_fabric::experimental::udm
