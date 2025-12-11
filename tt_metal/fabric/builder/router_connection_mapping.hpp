// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

/**
 * @brief Represents a single downstream connection target for a sender channel
 *
 * This struct defines where a sender channel should connect to, including:
 * - The connection type (INTRA_MESH, MESH_TO_Z, Z_TO_MESH)
 * - The target virtual channel (VC)
 * - The target sender channel index
 * - Optional target direction (for local connections)
 */
struct ConnectionTarget {
    ConnectionType type;
    uint32_t target_vc;
    uint32_t target_sender_channel;
    std::optional<RoutingDirection> target_direction;  // Used for MESH_TO_Z and Z_TO_MESH

    ConnectionTarget(
        ConnectionType type_,
        uint32_t target_vc_,
        uint32_t target_sender_channel_,
        std::optional<RoutingDirection> target_direction_ = std::nullopt)
        : type(type_),
          target_vc(target_vc_),
          target_sender_channel(target_sender_channel_),
          target_direction(target_direction_) {}
};

/**
 * @brief Key for identifying a sender channel (VC + channel index)
 */
struct SenderChannelKey {
    uint32_t vc;
    uint32_t sender_channel;

    bool operator<(const SenderChannelKey& other) const {
        if (vc != other.vc) return vc < other.vc;
        return sender_channel < other.sender_channel;
    }

    bool operator==(const SenderChannelKey& other) const {
        return vc == other.vc && sender_channel == other.sender_channel;
    }
};

/**
 * @brief Defines sender channel to downstream target mappings for a router
 *
 * This class encapsulates the connection logic for routers, mapping each sender
 * channel to its downstream connection targets. It supports:
 * - Mesh routers with 1D/2D topologies
 * - Mesh routers with MESH_TO_Z connections (when Z router present on device)
 * - Z routers with multi-target VC1 connections
 *
 * Key insight: Mapping is for SENDER channels, not receivers. A sender channel
 * may have multiple downstream targets (e.g., Z router VC1 connecting to 2-4 mesh routers).
 */
class RouterConnectionMapping {
public:
    RouterConnectionMapping() = default;

    /**
     * @brief Get downstream connection targets for a sender channel
     *
     * @param vc Virtual channel index
     * @param sender_channel Sender channel index within the VC
     * @return Vector of connection targets (may be empty if no connections)
     */
    std::vector<ConnectionTarget> get_downstream_targets(uint32_t vc, uint32_t sender_channel) const;

    /**
     * @brief Factory method for mesh router connection mapping
     *
     * Creates a mapping for a standard mesh router based on topology and direction.
     *
     * @param topology Mesh topology (1D or 2D)
     * @param direction Router's direction (NORTH, EAST, SOUTH, WEST)
     * @param has_z Whether this device has a Z router (enables MESH_TO_Z connections)
     * @return Configured RouterConnectionMapping for mesh router
     */
    static RouterConnectionMapping for_mesh_router(
        Topology topology,
        RoutingDirection direction,
        bool has_z);

    /**
     * @brief Factory method for Z router connection mapping
     *
     * Creates a mapping for a Z router with:
     * - VC0: Standard mesh forwarding (if applicable)
     * - VC1: Multi-target Z_TO_MESH connections (N/E/S/W intent)
     *
     * Note: Mapping specifies all 4 directions as intent. FabricBuilder
     * will skip non-existent directions based on device position (2-4 mesh routers).
     *
     * @return Configured RouterConnectionMapping for Z router
     */
    static RouterConnectionMapping for_z_router();

    /**
     * @brief Check if a sender channel has any downstream targets
     */
    bool has_targets(uint32_t vc, uint32_t sender_channel) const;

    /**
     * @brief Get total number of configured sender channels across all VCs
     */
    size_t get_total_sender_count() const { return sender_to_targets_.size(); }

    /**
     * @brief Get all sender channel keys (for iteration/testing)
     */
    std::vector<SenderChannelKey> get_all_sender_keys() const;

private:
    // Maps (VC, sender_channel) → list of downstream targets
    std::map<SenderChannelKey, std::vector<ConnectionTarget>> sender_to_targets_;

    /**
     * @brief Add a connection target for a sender channel
     */
    void add_target(uint32_t vc, uint32_t sender_channel, const ConnectionTarget& target);

    /**
     * @brief Helper to compute opposite direction for mesh routers
     */
    static RoutingDirection get_opposite_direction(RoutingDirection dir);
};

}  // namespace tt::tt_fabric
