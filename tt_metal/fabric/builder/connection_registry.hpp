// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/routing_table_generator.hpp>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

/**
 * ConnectionType - Categorizes the type of connection between routers
 */
enum class ConnectionType : std::uint8_t {
    INVALID,
    INTRA_MESH,  // Connection between mesh routers on different devices
    MESH_TO_Z,   // Connection from mesh router to Z router (same device)
    Z_TO_MESH,   // Connection from Z router to mesh router (same device)
};

/**
 * RouterConnectionRecord - Records a single connection between routers
 *
 * This structure captures all relevant information about a connection
 * for testing and validation purposes.
 */
struct RouterConnectionRecord {
    // Source router information
    FabricNodeId source_node;
    RoutingDirection source_direction = RoutingDirection::NONE;
    chan_id_t source_eth_chan = -1;
    uint32_t source_vc = -1;
    uint32_t source_receiver_channel = -1;

    // Destination router information
    FabricNodeId dest_node;
    RoutingDirection dest_direction = RoutingDirection::NONE;
    chan_id_t dest_eth_chan = -1;
    uint32_t dest_vc = -1;
    uint32_t dest_sender_channel = -1;

    // Connection metadata
    ConnectionType connection_type = ConnectionType::INVALID;
};

/**
 * ConnectionRegistry - Records all router connections for testing/validation
 *
 * This registry provides visibility into the connection graph between routers.
 * It's used during Phase 0 to add testing infrastructure and will be extended
 * in later phases to support Z router connections.
 *
 * Usage:
 *   auto registry = std::make_shared<ConnectionRegistry>();
 *   // Pass to builders during construction
 *   // Query connections for testing
 *   auto connections = registry->get_all_connections();
 */
class ConnectionRegistry {
public:
    ConnectionRegistry() = default;

    /**
     * Record a connection between two routers
     *
     * @param record The connection record to store
     */
    void record_connection(const RouterConnectionRecord& record);

    /**
     * Get all recorded connections
     *
     * @return Vector of all connection records
     */
    const std::vector<RouterConnectionRecord>& get_all_connections() const;

    /**
     * Get connections from a specific source router
     *
     * @param source_node The source fabric node
     * @param source_direction The source router direction
     * @return Vector of connections from this source
     */
    std::vector<RouterConnectionRecord> get_connections_from_source(
        FabricNodeId source_node,
        RoutingDirection source_direction) const;

    /**
     * Get connections to a specific destination router
     *
     * @param dest_node The destination fabric node
     * @param dest_direction The destination router direction
     * @return Vector of connections to this destination
     */
    std::vector<RouterConnectionRecord> get_connections_to_dest(
        FabricNodeId dest_node,
        RoutingDirection dest_direction) const;

    /**
     * Get connections of a specific type
     *
     * @param type The connection type to filter by
     * @return Vector of connections of the specified type
     */
    std::vector<RouterConnectionRecord> get_connections_by_type(ConnectionType type) const;

    /**
     * Get connections by source node (all routers on that node)
     *
     * @param source_node The source fabric node
     * @return Vector of connections from this node
     */
    std::vector<RouterConnectionRecord> get_connections_by_source_node(FabricNodeId source_node) const;

    /**
     * Get connections by destination node (all routers on that node)
     *
     * @param dest_node The destination fabric node
     * @return Vector of connections to this node
     */
    std::vector<RouterConnectionRecord> get_connections_by_dest_node(FabricNodeId dest_node) const;

    /**
     * Clear all recorded connections
     */
    void clear();

    /**
     * Get the total number of recorded connections
     *
     * @return Number of connections
     */
    size_t size() const;

private:
    std::vector<RouterConnectionRecord> connections_;
};

}  // namespace tt::tt_fabric
