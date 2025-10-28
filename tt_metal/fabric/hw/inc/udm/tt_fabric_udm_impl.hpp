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
#include "dev_mem_map.h"

namespace tt::tt_fabric::udm {

/**
 * @brief UDM (Unified Data Movement) fields for fabric write operations
 *
 * Contains the source information needed for UDM write operations:
 * - src_noc_xy: Source NOC coordinates (X in lower byte, Y in upper byte)
 * - risc_id: Processor ID (BRISC, NCRISC, etc.)
 * - trid: Transaction ID for tracking operations
 */
struct udm_write_fields {
    uint16_t src_noc_xy;
    uint8_t risc_id;
    uint16_t trid;
};

// Helper template for static_assert
template <typename T>
inline constexpr bool always_false_v = false;

/**
 * @brief Set unicast route for UDMHybridMeshPacketHeader with UDM fields
 *
 * Configures routing and UDM control fields for hybrid mesh packet headers.
 * This overload handles 2D mesh routing with UDM-specific metadata.
 *
 * @param packet_header Pointer to the UDM hybrid mesh packet header
 * @param udm UDM fields containing source information
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
    packet_header->udm_control.write.src_noc_xy = udm.src_noc_xy;
    packet_header->udm_control.write.risc_id = udm.risc_id;
    packet_header->udm_control.write.transaction_id = udm.trid;
    return result;
}

/**
 * @brief Helper function to set unicast routing based on packet header type
 *
 * This function provides a unified interface for setting up fabric unicast routing
 * across different packet header types (UDMHybridMeshPacketHeader, UDMLowLatencyPacketHeader).
 * It automatically selects the appropriate routing API based on the packet header type.
 *
 * @tparam PACKET_HEADER_TYPE The type of packet header (auto-deduced)
 * @param packet_header Pointer to the packet header
 * @param dst_dev_id Destination device ID
 * @param dst_mesh_id Destination mesh ID
 * @param trid Transaction ID for UDM operations
 */
template <typename PACKET_HEADER_TYPE>
FORCE_INLINE void fabric_write_set_unicast_route(
    volatile tt_l1_ptr PACKET_HEADER_TYPE* packet_header, uint16_t dst_dev_id, uint16_t dst_mesh_id, uint16_t trid) {
    uint16_t my_noc_xy = my_x[edm_to_local_chip_noc] | my_y[edm_to_local_chip_noc] << 8;

    // Create udm_write_fields struct with the necessary fields
    udm_write_fields udm = {my_noc_xy, proc_type, trid};

    if constexpr (std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMHybridMeshPacketHeader>) {
        fabric_write_set_unicast_route(packet_header, udm, dst_dev_id, dst_mesh_id);
    } else if constexpr (std::is_same_v<PACKET_HEADER_TYPE, tt::tt_fabric::UDMLowLatencyPacketHeader>) {
        fabric_write_set_unicast_route(packet_header, udm, dst_dev_id);
    } else {
        static_assert(always_false_v<PACKET_HEADER_TYPE>, "Unsupported PACKET_HEADER_TYPE for fabric routing");
    }
}

/**
 * @brief Set unicast route for UDMLowLatencyPacketHeader with UDM fields
 *
 * Configures routing and UDM control fields for low latency packet headers.
 * This overload handles 1D routing with UDM-specific metadata.
 *
 * @param packet_header Pointer to the UDM low latency packet header
 * @param udm UDM fields containing source information
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
    packet_header->udm_control.write.src_noc_xy = udm.src_noc_xy;
    packet_header->udm_control.write.risc_id = udm.risc_id;
    packet_header->udm_control.write.transaction_id = udm.trid;

    return result;
}

/**
 * @brief Get or create a singleton fabric connection for the current worker
 *
 * This function manages a static connection instance that persists across calls.
 * The connection is initialized on first use and reused for subsequent operations.
 *
 * @return Reference to the fabric connection
 */
inline tt::tt_fabric::WorkerToFabricEdmSender& get_fabric_connection() {
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

}  // namespace tt::tt_fabric::udm
