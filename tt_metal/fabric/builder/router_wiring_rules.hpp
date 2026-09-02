// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

// Forward declaration
struct IntermeshVCConfig;

/**
 * Pure router-archetype rules over topology, facing, capabilities, Z-port role, express mode, and
 * VC config. Per-router wiring, family maxima, and injection flags share these derivations.
 */

/**
 * @brief The turn-matrix primitive: does the router facing `producer_direction` wire into the
 * router facing `egress_direction` on the same chip?
 *
 * Turn sets, producer arity, and injection guards all use this relation. U-turns are never wired.
 * In express mode, a same-mesh X producer cannot turn into a protected Y egress, while an intermesh
 * landing can begin Y. Z egress requires a local Z role. Non-express mode preserves cardinal wiring
 * and targets Z only for an intermesh boundary. Boundary producers wire only on VC1; all other
 * producer roles are VC-independent.
 */
bool wires_into(
    RoutingDirection producer_direction,
    EdgeCapability producer_capability,
    RoutingDirection egress_direction,
    std::optional<EdgeCapability> egress_capability,
    ZPortRole chip_z_role,
    bool express_routing_enabled,
    uint32_t vc);

// Opposite direction for mesh routers (N<->S, E<->W). Z has no opposite.
RoutingDirection get_opposite_direction(RoutingDirection dir);

/**
 * @brief The canonical express-endpoint chip: every cardinal intramesh, Z is the chord
 *
 * Input for the express family maximum; no per-chip capability set produces a wider router.
 */
PerDirectionCapabilities canonical_express_endpoint_capabilities();

/**
 * @brief VC0 sender slots a router facing `direction` needs on a chip with these capabilities
 *
 * Counts the local worker plus wired producers. Per-chip callers must pass actual capabilities;
 * canonical_express_endpoint_capabilities() is only for the family maximum.
 */
uint32_t express_vc0_producer_arity(RoutingDirection direction, const PerDirectionCapabilities& caps);

/**
 * @brief Uniform VC0/VC1 sender counts for the express family
 *
 * Family maxima over facing directions on the canonical endpoint. Individual routers fill a
 * subset; channel trimming narrows from actual per-chip capabilities.
 */
uint32_t express_vc0_sender_count();
uint32_t express_vc1_sender_count();

/**
 * Non-express and boundary family maxima derived from the 2D direction count. Express maxima are
 * derived above by iterating the canonical endpoint's facings.
 */

// Non-express forwarding family: the worker plus every non-self cardinal producer.
constexpr uint32_t non_express_vc0_sender_count() { return 1 + (builder_config::num_mesh_directions_2d - 1); }

// Keep the wiring derivation equal to the tensix/L1-domain constant.
static_assert(
    non_express_vc0_sender_count() == builder_config::num_sender_channels_2d_mesh,
    "The frozen non-express forwarding VC0 count and num_sender_channels_2d_mesh must stay equal");

// Non-express forwarding VC1: the non-self cardinal producers (no worker on VC1).
constexpr uint32_t non_express_vc1_sender_count() { return builder_config::num_mesh_directions_2d - 1; }

// Boundary family: the max is attained by a non-express boundary chip, whose every
// mesh-direction producer wires into the boundary egress.
constexpr uint32_t boundary_vc0_sender_count() { return 1 + builder_config::num_mesh_directions_2d; }

// The from-boundary fanout width: every mesh direction.
constexpr uint32_t boundary_vc1_sender_count() { return builder_config::num_mesh_directions_2d; }

/**
 * Complete per-VC sender/receiver counts and flat-index bases for one router. Derived from the same
 * facts as the connection map rather than recovered from map entries at consumption sites.
 */
struct RouterVcShape {
    uint32_t num_vcs = 0;

    // Bases are prefix sums of lower-VC channel counts.
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_counts{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_flat_base{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receiver_counts{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receiver_flat_base{};

    // Bounds-checked (vc, channel) to flat-index conversion.
    uint32_t flat_sender_id(uint32_t vc, uint32_t channel) const {
        TT_FATAL(
            vc < builder_config::MAX_NUM_VCS && channel < sender_counts[vc],
            "No sender channel {} on VC{} (router has {})",
            channel,
            vc,
            vc < builder_config::MAX_NUM_VCS ? sender_counts[vc] : 0);
        return sender_flat_base[vc] + channel;
    }

    uint32_t flat_receiver_id(uint32_t vc, uint32_t channel) const {
        TT_FATAL(
            vc < builder_config::MAX_NUM_VCS && channel < receiver_counts[vc],
            "No receiver channel {} on VC{} (router has {})",
            channel,
            vc,
            vc < builder_config::MAX_NUM_VCS ? receiver_counts[vc] : 0);
        return receiver_flat_base[vc] + channel;
    }
};

/**
 * @brief The one per-VC shape derivation for any router
 *
 * Topology gates channel counts, while num_vcs remains config-only: a 1D router may report VC1
 * with zero VC1 channels. The derivation emits all flat bases and enforces the channel ceilings.
 * Reading facing capability and Z role from one capability set prevents inconsistent inputs.
 */
RouterVcShape router_vc_shape(
    Topology topology,
    RoutingDirection facing,
    const PerDirectionCapabilities& chip_capabilities,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config);

/**
 * @brief Represents a single downstream connection target for a receiver channel
 *
 * Identifies a local turn by direction and VC. Establishment computes the sender slot from the
 * direction-to-slot bijection; edge capability identifies intermesh boundary turns.
 */
struct ConnectionTarget {
    uint32_t target_vc;
    std::optional<RoutingDirection> target_direction;  // The downstream direction this target reaches

    ConnectionTarget(uint32_t target_vc_, std::optional<RoutingDirection> target_direction_ = std::nullopt) :
        target_vc(target_vc_), target_direction(target_direction_) {}
};

/**
 * Downstream turns fed by the router's receiver on each VC. One receiver exists per active VC;
 * an empty vector means that VC has no wiring.
 */
using RouterTurnSet = std::array<std::vector<ConnectionTarget>, builder_config::MAX_NUM_VCS>;

/**
 * @brief The one turn-set derivation for every router
 *
 * Boundary routers use the VC1 boundary template. Other routers derive turns through wires_into():
 * 1D uses the opposite direction, non-express 2D uses non-self cardinals, and express mode applies
 * its dimension-order and Z-role rules. Shape and turn derivations consume the same capability and
 * VC configuration objects so callers cannot describe one router inconsistently.
 */
RouterTurnSet turn_set_for_router(
    Topology topology,
    RoutingDirection facing,
    const PerDirectionCapabilities& chip_capabilities,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config);

/**
 * Channel shape and turn set derived from one fact tuple. Use this when both results are needed so
 * their inputs cannot diverge.
 */
struct RouterArchetype {
    RouterVcShape shape;
    RouterTurnSet turns;
};

RouterArchetype router_archetype(
    Topology topology,
    RoutingDirection facing,
    const PerDirectionCapabilities& chip_capabilities,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config);

/**
 * Datamover owner for sender channels on one VC. In MUX mode TENSIX owns VC0; ERISC owns VC1/VC2.
 * The link-scoped downstream owner is supplied at the use site rather than stored in the archetype.
 */
enum class BuilderType : uint8_t {
    ERISC = 0,
    TENSIX = 1,
};

inline BuilderType builder_type_for_vc(uint32_t vc, bool downstream_is_tensix_builder) {
    return (vc == 0 && downstream_is_tensix_builder) ? BuilderType::TENSIX : BuilderType::ERISC;
}

}  // namespace tt::tt_fabric
