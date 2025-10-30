// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// This file contains implementation details and types for UDM fabric functions.
// It is included at the top of tt_fabric_udm.h

// Required headers for the implementations
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "dev_mem_map.h"
#include <type_traits>

namespace tt::tt_fabric::udm {

/**
 * @brief Enum for UDM control fields
 */
enum class UDM_CONTROL_FIELD { POSTED, SRC_CHIP_ID, SRC_MESH_ID, SRC_NOC_X, SRC_NOC_Y, RISC_ID, TRANSACTION_ID };

/**
 * @brief UDM (Unified Data Movement) fields for fabric write operations
 *
 * Contains the source information needed for UDM write operations:
 * - src_noc_x: Source NOC X coordinate
 * - src_noc_y: Source NOC Y coordinate
 * - risc_id: Processor ID (BRISC, NCRISC, etc.)
 * - trid: Transaction ID for tracking operations
 * - posted: Whether this is a posted write (1) or non-posted write (0)
 */
struct udm_write_fields {
    uint8_t src_noc_x;
    uint8_t src_noc_y;
    uint8_t risc_id;
    uint16_t trid;
    uint8_t posted;
};

/**
 * @brief Set unicast route for UDMLowLatencyPacketHeader with UDM fields
 *
 * Configures routing and UDM control fields for low latency packet headers.
 * This overload handles 1D routing with UDM-specific metadata.
 *
 * @param packet_header Pointer to the UDM low latency packet header
 * @param udm UDM fields containing source information (including posted flag)
 * @param dst_dev_id Destination device ID
 * @return true if route was successfully set, false otherwise
 */
inline bool fabric_write_set_unicast_route(
    volatile tt_l1_ptr UDMLowLatencyPacketHeader* packet_header, udm_write_fields& udm, uint16_t dst_dev_id) {
    tt_l1_ptr tensix_routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr tensix_routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);

    // First call the original API by casting to base type
    // Use the template parameters to call the correct overload
    bool result =
        fabric_set_unicast_route(static_cast<volatile tt_l1_ptr LowLatencyPacketHeader*>(packet_header), dst_dev_id);

    // Set UDM control fields for write operations using routing table info
    packet_header->udm_control.write.src_chip_id = routing_table->my_device_id;
    packet_header->udm_control.write.src_mesh_id = routing_table->my_mesh_id;
    packet_header->udm_control.write.src_noc_x = udm.src_noc_x;
    packet_header->udm_control.write.src_noc_y = udm.src_noc_y;
    packet_header->udm_control.write.risc_id = udm.risc_id;
    packet_header->udm_control.write.transaction_id = udm.trid;
    packet_header->udm_control.write.posted = udm.posted;

    return result;
}

/**
 * @brief Set unicast route for UDMHybridMeshPacketHeader with UDM fields
 *
 * Configures routing and UDM control fields for hybrid mesh packet headers.
 * This overload handles 2D mesh routing with UDM-specific metadata.
 *
 * @param packet_header Pointer to the UDM hybrid mesh packet header
 * @param udm UDM fields containing source information (including posted flag)
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @return true if route was successfully set, false otherwise
 */
inline bool fabric_write_set_unicast_route(
    volatile tt_l1_ptr UDMHybridMeshPacketHeader* packet_header,
    udm_write_fields& udm,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id) {
    tt_l1_ptr tensix_routing_l1_info_t* routing_table =
        reinterpret_cast<tt_l1_ptr tensix_routing_l1_info_t*>(MEM_TENSIX_ROUTING_TABLE_BASE);

    // First call the original API by casting to base type
    bool result = fabric_set_unicast_route(
        static_cast<volatile tt_l1_ptr HybridMeshPacketHeader*>(packet_header), dst_dev_id, dst_mesh_id);

    // Set UDM control fields for write operations using routing table info
    packet_header->udm_control.write.src_chip_id = routing_table->my_device_id;
    packet_header->udm_control.write.src_mesh_id = routing_table->my_mesh_id;
    packet_header->udm_control.write.src_noc_x = udm.src_noc_x;
    packet_header->udm_control.write.src_noc_y = udm.src_noc_y;
    packet_header->udm_control.write.risc_id = udm.risc_id;
    packet_header->udm_control.write.transaction_id = udm.trid;
    packet_header->udm_control.write.posted = udm.posted;
    return result;
}

/**
 * @brief Helper function to set unicast routing based on packet header type
 *
 * This function provides a unified interface for setting up fabric unicast routing.
 * It creates the necessary UDM fields and calls the appropriate routing function
 * based on the packet header type defined at compile time.
 *
 * @param packet_header Pointer to the packet header
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @param trid Transaction ID for UDM operations
 * @param posted Whether this is a posted write (1) or non-posted write (0, default)
 */
template <typename T>
FORCE_INLINE void fabric_write_set_unicast_route_impl(
    volatile tt_l1_ptr T* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t trid, uint8_t posted) {
    udm_write_fields udm = {my_x[edm_to_local_chip_noc], my_y[edm_to_local_chip_noc], proc_type, trid, posted};

    if constexpr (std::is_same_v<T, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
        fabric_write_set_unicast_route(packet_header, udm, dst_dev_id, dst_mesh_id);
    } else if constexpr (std::is_same_v<T, tt::tt_fabric::UDMLowLatencyPacketHeader>) {
        fabric_write_set_unicast_route(packet_header, udm, dst_dev_id);
    } else {
        // Compile error for unsupported types
        static_assert(
            std::is_same_v<T, tt::tt_fabric::UDMHybridMeshPacketHeader> ||
                std::is_same_v<T, tt::tt_fabric::UDMLowLatencyPacketHeader>,
            "Unsupported packet header type for fabric_write_set_unicast_route_impl");
    }
}

FORCE_INLINE void fabric_write_set_unicast_route(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header,
    uint16_t dst_dev_id,
    uint16_t dst_mesh_id,
    uint16_t trid,
    uint8_t posted) {
    fabric_write_set_unicast_route_impl(packet_header, dst_dev_id, dst_mesh_id, trid, posted);
}

/**
 * @brief Set individual UDM control fields in the packet header
 *
 * This template function allows setting specific UDM control fields individually
 * without affecting other fields. It provides fine-grained control over the
 * packet header configuration.
 *
 * @tparam Field The control field to set (from UDM_CONTROL_FIELD enum)
 * @tparam T The packet header type (auto-deduced)
 * @tparam V The value type (auto-deduced)
 * @param packet_header Pointer to the packet header (UDMLowLatencyPacketHeader or UDMHybridMeshPacketHeader)
 * @param value The value to set for the specified field
 *
 */
template <UDM_CONTROL_FIELD Field, typename T, typename V>
FORCE_INLINE void fabric_write_set_unicast_route_control_field(volatile tt_l1_ptr T* packet_header, V value) {
    // Ensure the packet header type has UDM control fields
    static_assert(
        std::is_same_v<T, UDMLowLatencyPacketHeader> || std::is_same_v<T, UDMHybridMeshPacketHeader> ||
            std::is_same_v<T, PACKET_HEADER_TYPE>,
        "Packet header must be a UDM packet header type");

    if constexpr (Field == UDM_CONTROL_FIELD::POSTED) {
        packet_header->udm_control.write.posted = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_CHIP_ID) {
        packet_header->udm_control.write.src_chip_id = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_MESH_ID) {
        packet_header->udm_control.write.src_mesh_id = static_cast<uint16_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_NOC_X) {
        packet_header->udm_control.write.src_noc_x = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::SRC_NOC_Y) {
        packet_header->udm_control.write.src_noc_y = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::RISC_ID) {
        packet_header->udm_control.write.risc_id = static_cast<uint8_t>(value);
    } else if constexpr (Field == UDM_CONTROL_FIELD::TRANSACTION_ID) {
        packet_header->udm_control.write.transaction_id = static_cast<uint8_t>(value);
    }
}

/**
 * @brief Get or create a singleton fabric connection for the current worker
 *
 * This function manages a static connection instance that persists across calls.
 * The connection is initialized on first use and reused for subsequent operations.
 *
 * @return Reference to the fabric connection
 */
inline tt::tt_fabric::WorkerToFabricEdmSender& get_or_open_fabric_connection() {
    static tt::tt_fabric::WorkerToFabricEdmSender* connection = nullptr;
    static bool initialized = false;

    if (!initialized) {
        // Build connection from runtime args on first use
        // This assumes the runtime args are set up properly by the host
        // TODO: instead of using rt args, use the reserved L1 region for get the correct ETH channel, and semaphore
        // addresses.
        size_t rt_args_idx = 0;
        static tt::tt_fabric::WorkerToFabricEdmSender conn;
        conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(rt_args_idx);
        conn.open();
        connection = &conn;
        initialized = true;
    }

    return *connection;
}

/**
 * @brief Get or allocate a packet header for UDM operations
 *
 * Returns a singleton packet header for the current processor.
 * If a header has already been allocated, returns the existing one.
 * Otherwise allocates a new header from the PacketHeaderPool.
 *
 * This ensures we reuse the same header location for all operations,
 * similar to the get_fabric_connection pattern. This is more efficient
 * than allocating and releasing headers for each operation.
 *
 * @return Pointer to the allocated packet header
 */
inline volatile tt_l1_ptr PACKET_HEADER_TYPE* get_or_allocate_header() {
    static volatile tt_l1_ptr PACKET_HEADER_TYPE* singleton_header = nullptr;
    if (singleton_header == nullptr) {
        singleton_header = PacketHeaderPool::allocate_header();
    }
    return singleton_header;
}

}  // namespace tt::tt_fabric::udm
