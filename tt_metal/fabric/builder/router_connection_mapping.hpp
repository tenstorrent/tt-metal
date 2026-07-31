// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

/**
 * @brief Represents a single downstream connection target for a receiver channel
 *
 * This struct defines where a receiver channel should connect to, including:
 * - The target virtual channel (VC)
 * - Optional target direction (for local connections)
 *
 * Targets carry no connection type and no channel index: a local turn is identified by its
 * direction and VC, the slot a producer is placed into is computed at establishment time from the
 * direction<->slot bijection (get_downstream_sender_channel_for_vc), and the boundary turn to an
 * intermesh edge is identified by the edge's capability, not a type label.
 */
struct ConnectionTarget {
    uint32_t target_vc;
    std::optional<RoutingDirection> target_direction;  // The downstream direction this target reaches

    ConnectionTarget(uint32_t target_vc_, std::optional<RoutingDirection> target_direction_ = std::nullopt) :
        target_vc(target_vc_), target_direction(target_direction_) {}
};

/**
 * @brief The per-VC turn table of one router: which downstream routers its receivers feed
 *
 * Maps each VC to the list of downstream turns (direction + target VC) out of this router's
 * receiver on that VC. The table is keyed by VC alone: a router services exactly one receiver
 * per active VC by construction (router_vc_shape gives it one), so there is no receiver
 * dimension to index.
 *
 * The wiring rules the table is built from -- the wires_into turn matrix, the router_vc_shape
 * derivation, and the arity counts -- live in builder/router_wiring_rules.* as free functions.
 */
class RouterConnectionMapping {
public:
    RouterConnectionMapping() = default;

    /**
     * @brief Get downstream connection targets on a VC
     *
     * @param vc Virtual channel index
     * @return Vector of connection targets (empty if none are wired on that VC)
     */
    std::vector<ConnectionTarget> get_downstream_targets(uint32_t vc) const;

    /**
     * @brief The one connection-map factory for every router
     *
     * Behaviour is keyed on what role the ports have, not which ports they are:
     * - A port with no routing direction (facing Z, capability INTERMESH) gets the boundary
     *   template: the full non-self set on VC1, typed from-boundary. Its VC0 senders are fed by
     *   the mesh routers' boundary targets on their own maps.
     * - A routing-direction port gets its turn set from the wires_into primitive: 1D is the
     *   opposite direction; legacy 2D is every non-self cardinal; express adds the express rule
     *   (an intramesh X ingress unwires from intramesh Y, a landing X ingress does not).
     * - The chip's extra port enters the set only when it has one: an express chord is an ordinary
     *   same-VC target; an intermesh boundary is reached through the boundary target on VC0 (and on
     *   VC1 only in pass-through mode); nothing exists without the port.
     *
     * @param topology Mesh topology (1D or 2D)
     * @param facing This router's own direction
     * @param edge_capability Capability of this router's own edge
     * @param z_role What this chip's extra port is used for (chord, boundary, or none)
     * @param express_routing_enabled Mesh-level: express chords are materialized and validated
     * @param enable_vc1 Whether VC1 (inter-mesh) connections should be created
     * @param enable_mesh_pass_through EXPERIMENTAL: also forwards VC1 traffic to the local
     *        intermesh Z boundary (the boundary target on VC1) so inter-mesh traffic can pass
     *        through this mesh (A->B->C). Reuses VC1; not deadlock-safe.
     * @return Configured RouterConnectionMapping for the router
     */
    static RouterConnectionMapping for_router(
        Topology topology,
        RoutingDirection facing,
        EdgeCapability edge_capability = EdgeCapability::INTRAMESH_CARDINAL,
        ZPortRole z_role = ZPortRole::NONE,
        bool express_routing_enabled = false,
        bool enable_vc1 = false,
        bool enable_mesh_pass_through = false);

    /**
     * @brief Check if a VC has any downstream targets
     */
    bool has_targets(uint32_t vc) const;

    /**
     * @brief Get total number of downstream targets across all VCs
     */
    size_t get_total_target_count() const;

private:
    // Per-VC turn lists, indexed by VC. Every arm of for_router pushes onto an existing slot, so
    // an empty vector and an absent key are the same state: nothing is wired on that VC.
    std::array<std::vector<ConnectionTarget>, builder_config::MAX_NUM_VCS> targets_by_vc_{};

    /**
     * @brief Add a connection target on a VC
     */
    void add_target(uint32_t vc, const ConnectionTarget& target);
};

}  // namespace tt::tt_fabric
