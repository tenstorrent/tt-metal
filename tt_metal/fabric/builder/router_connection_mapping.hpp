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
 * @brief Represents a single downstream connection target for a receiver channel
 *
 * This struct defines where a receiver channel should connect to, including:
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
 * @brief Key for identifying a receiver channel (VC + channel index)
 */
struct ReceiverChannelKey {
    uint32_t vc;
    uint32_t receiver_channel;

    bool operator<(const ReceiverChannelKey& other) const {
        if (vc != other.vc) {
            return vc < other.vc;
        }
        return receiver_channel < other.receiver_channel;
    }

    bool operator==(const ReceiverChannelKey& other) const {
        return vc == other.vc && receiver_channel == other.receiver_channel;
    }
};

/**
 * @brief Defines receiver channel to downstream target mappings for a router
 *
 * This class encapsulates the connection logic for routers, mapping each receiver
 * channel to its downstream connection targets. It supports:
 * - Mesh routers with 1D/2D topologies
 * - Mesh routers with MESH_TO_Z connections (when Z router present on device)
 * - Z routers with multi-target VC1 connections
 */
class RouterConnectionMapping {
public:
    RouterConnectionMapping() = default;

    /**
     * @brief Get downstream connection targets for a receiver channel
     *
     * @param vc Virtual channel index
     * @param receiver_channel Sender channel index within the VC
     * @return Vector of connection targets (may be empty if no connections)
     */
    std::vector<ConnectionTarget> get_downstream_targets(uint32_t vc, uint32_t receiver_channel) const;

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
        Topology topology, RoutingDirection direction, bool has_z, bool enable_vc1 = false);

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
     * @brief Check if a receiver channel has any downstream targets
     */
    bool has_targets(uint32_t vc, uint32_t receiver_channel) const;

    /**
     * @brief Get total number of configured sender channels across all VCs
     */
    size_t get_total_sender_count() const { return std::accumulate(receiver_to_targets_.begin(), receiver_to_targets_.end(), 0, [](size_t sum, const auto& pair) { return sum + pair.second.size(); }); }

    /**
     * @brief Get all receiver channel keys (for iteration/testing)
     */
    std::vector<ReceiverChannelKey> get_all_receiver_keys() const;

private:
    // Maps (VC, sender_channel) → list of downstream targets
    std::map<ReceiverChannelKey, std::vector<ConnectionTarget>> receiver_to_targets_;

    /**
     * @brief Add a connection target for a receiver channel
     */
    void add_target(uint32_t vc, uint32_t receiver_channel, const ConnectionTarget& target);

    /**
     * @brief Helper to compute opposite direction for mesh routers
     */
    static RoutingDirection get_opposite_direction(RoutingDirection dir);
};

}  // namespace tt::tt_fabric
