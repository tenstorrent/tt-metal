// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <vector>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
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
     * @param enable_vc1 Whether VC1 (inter-mesh) connections should be created
     * @param enable_mesh_pass_through EXPERIMENTAL: when set (and has_z), also forwards VC1 traffic to
     *        the local Z router (MESH_TO_Z on VC1) so inter-mesh traffic can pass through this mesh
     *        toward a further mesh (A->B->C) instead of sinking here. Reuses VC1; not deadlock-safe.
     * @param express_routing_enabled When set, build the express transition set instead: cardinal and
     *        express Z outputs on every carrier VC, with an ordinary X ingress unwired from intramesh
     *        Y egress so dimension order holds. Non-express wiring is left byte-for-byte as it was.
     * @param ingress_capability Capability of this router's own edge. Only consulted under express
     *        routing, where an INTERMESH landing on an E/W port stays eligible to begin Y.
     * @param has_intramesh_express Whether this chip terminates a same-mesh express chord. Only
     *        consulted under express routing: a Z output is emitted only when the chord exists. On a
     *        chip whose only Z edge crosses a mesh boundary, a Z target would resolve to the
     *        intermesh Z router and leak same-mesh traffic onto the boundary link.
     * @return Configured RouterConnectionMapping for mesh router
     */
    static RouterConnectionMapping for_mesh_router(
        Topology topology,
        RoutingDirection direction,
        bool has_z,
        bool enable_vc1 = false,
        bool enable_mesh_pass_through = false,
        bool express_routing_enabled = false,
        EdgeCapability ingress_capability = EdgeCapability::INTRAMESH_CARDINAL,
        bool has_intramesh_express = false);

    /**
     * @brief Legal outbound directions for one express-routing mesh router
     *
     * Exposed for regression: the transition set is the whole point of the express wiring, so it is
     * checked directly rather than only through the assembled mapping.
     */
    static std::vector<RoutingDirection> express_outbound_directions(
        RoutingDirection direction, EdgeCapability ingress_capability);

    /**
     * @brief The outbound directions this chip's router actually wires under express routing
     *
     * The legal transitions from express_outbound_directions minus any output whose edge does not
     * exist here: a Z output requires the chip to terminate an intramesh express chord. On a chip
     * whose only Z edge crosses a mesh boundary, a Z target would resolve to the intermesh Z router
     * and leak same-mesh traffic onto the boundary link.
     *
     * for_mesh_router and the injection-flag derivation both consume this, so the wired producer
     * set and the connection map cannot drift apart.
     */
    static std::vector<RoutingDirection> wired_express_outbound_directions(
        RoutingDirection direction, EdgeCapability ingress_capability, bool has_intramesh_express);

    /**
     * @brief Would the router facing `producer_direction` (on the same chip) wire into the router
     * facing `egress_direction` under express routing?
     *
     * True exactly when `egress_direction` is in the producer's wired outbound set. The
     * injection-flag derivation uses this to classify only producers the connection map actually
     * wired -- in particular, an intramesh X producer is never wired into an intramesh Y egress
     * (dimension order), and no producer is wired into a Z egress on a chord-less chip.
     */
    static bool is_express_producer_wired(
        RoutingDirection producer_direction,
        EdgeCapability producer_capability,
        RoutingDirection egress_direction,
        bool has_intramesh_express);

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

    /**
     * @brief Add the MESH_TO_Z targets that reach a local intermesh Z router
     */
    static void add_mesh_to_z_targets(
        RouterConnectionMapping& mapping, Topology topology, bool enable_vc1, bool enable_mesh_pass_through);
};

}  // namespace tt::tt_fabric
