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
 * The wiring rules for one router, as free functions: the turn-matrix primitive everything is
 * built from, the per-VC channel shape derived from it, the arity/count derivations between
 * the two, and the per-VC turn set built from the primitive. These are facts about a router
 * archetype -- pure functions of (topology, facing, capability, ZPortRole, express, VC config),
 * carrying no eth channel -- shared by the per-router build and establishment
 * (compute_mesh_router_builder.cpp), the fabric-wide max-counts pass (fabric_builder_context.cpp),
 * and the injection-flag derivation, so none of them can drift from the others.
 */

/**
 * @brief The turn-matrix primitive: does the router facing `producer_direction` wire into the
 * router facing `egress_direction` on the same chip?
 *
 * The one place the wiring rule lives. Turn sets, producer arity, and the injection-flag
 * derivation are all built from this single relation, so the connection map and the guard
 * derivation cannot drift apart -- there is no set form of the rule for them to disagree with.
 *
 * The rule keys on the producer port's role -- the reduction of its (direction, capability)
 * pair, computed from both, substituting for neither:
 * - UNRESTRICTED (a Y cardinal port of any capability, or an X landing): no dimension-order
 *   limit; wires into every non-self egress the gates below admit.
 * - X_RING_ONLY (a same-mesh X port): dimension order; may not turn back into an INTRAMESH Y
 *   egress. The egress side is capability-keyed too (contract section 4.4), so a mesh seam
 *   stays wired whatever compass letter it sits on: leaving the mesh is not a turn back into
 *   the Y rings the ordering protects.
 * - EXPRESS_CHORD (Z carrying a same-mesh chord): a Y resource, like N/S, so
 *   likewise unrestricted; its feed rides every carrier VC.
 * - BOUNDARY (Z crossing a mesh boundary): carries no routing direction, so it
 *   sits outside the turn matrix; only the VC fact below applies to it.
 *
 * Two gates apply on the egress side. The mode gate: non-express is producer-blind -- every
 * non-self cardinal wires in, whatever the producer's role. The Z-existence gate: a Z egress is
 * wired only when the chip's Z port has a role at all (chord or boundary), and in
 * non-express mode only as the boundary template's target. A U-turn is never wired, whatever
 * the ports are.
 *
 * The VC matters exactly once: for a boundary producer (a Z-facing router whose edge is
 * INTERMESH). Its VC1 receiver fans out to every non-self VC1 sender (wired), while its VC0
 * receiver crosses over onto downstream VC1 senders and feeds nothing on VC0 (not wired). So
 * the boundary-producer arm answers `vc == 1`, in either mode -- that is a physical fact about
 * the boundary's receivers, not an express-mode rule. Every other producer is VC-agnostic on
 * this question, and turn_set_for_router may pass any VC when building a turn set, since that
 * arm is unreachable there (the boundary path early-returns before consulting this primitive).
 *
 * Guard classification follows from the answers: a cardinal producer into a boundary egress is
 * NON_RING (the egress is not a protected ring), and a boundary producer into a protected
 * cardinal egress on VC1 is ENTER -- the correct landing acquisition. The first arm is not
 * "NON_RING either way": only the second depends on the VC.
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
 * This is the capability set the express family-max count is evaluated against. It attains
 * the structural ceiling: every Y and X producer wires into an E/W facing under any capability
 * assignment, so no per-chip set produces a wider router.
 */
PerDirectionCapabilities canonical_express_endpoint_capabilities();

/**
 * @brief VC0 sender slots a router facing `direction` needs on a chip with these capabilities
 *
 * The local worker plus the producers the connection map wires into it. Arity depends on
 * facing: an E/W-facing router is fed by every Y producer (the Y->X turn is legal), while
 * dimension order leaves N/S/Z-facing routers with only their Y producers. INTERMESH
 * (landing-capable) directions widen their non-self set, so per-chip callers must pass the
 * actual chip's capabilities; the canonical endpoint set is only the family-max input.
 */
uint32_t express_vc0_producer_arity(RoutingDirection direction, const PerDirectionCapabilities& caps);

/**
 * @brief Uniform VC0/VC1 sender counts for the express family
 *
 * The family max over facing directions of wired-producer arity on the canonical endpoint
 * chip: one flat index space per family, with per-router wiring filling a subset and
 * per-direction channel trimming as the separate L1 lever for narrowing (which evaluates
 * arity against the actual per-chip capability set, not the canonical one).
 */
uint32_t express_vc0_sender_count();
uint32_t express_vc1_sender_count();

/**
 * The non-express and boundary family counts, as constexpr derivations from the 2D
 * mesh-direction count, stated next to the wiring rule that produces them. (The express
 * family's counts, declared above, are runtime derivations over the canonical endpoint chip --
 * it attains the structural ceiling, so they iterate facings.) Family counts are family MAXIMA
 * (per-chip narrowing is channel trimming's job, not the shape's). The non-express forwarding
 * counts are frozen by standing decision: byte-identical on every existing 2D configuration.
 */

// Non-express forwarding family: the worker plus every non-self cardinal producer.
constexpr uint32_t non_express_vc0_sender_count() { return 1 + (builder_config::num_mesh_directions_2d - 1); }

// The frozen count and the tensix/L1-domain constant are two sources for one number, and the
// include direction (rules -> config) forbids the config from calling the derivation. This is
// the compile-time tie between them; it becomes a single source when the tensix path is widened.
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
 * The complete per-VC channel shape of one router: how many sender and receiver channels it
 * has on each VC, where each VC starts in the flat index space, and how many VCs exist.
 * Computed ONCE from the same facts the connection map reads, so the count and every flat base
 * are facts upstream of layout -- never recovered by counting map entries, and never
 * recomputed at a consumption site (which is the class of bug that produced flat-9 aliasing).
 */
struct RouterVcShape {
    uint32_t num_vcs = 0;

    // Per-VC channel counts and their flat-index prefix sums, emitted by the derivation.
    // sender_flat_base[vc] = sum of sender_counts over lower VCs; receiver likewise. The VC2
    // receiver index that used to be "1 + num_vc1_receivers" is receiver_flat_base[2].
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_counts{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> sender_flat_base{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receiver_counts{};
    std::array<uint32_t, builder_config::MAX_NUM_VCS> receiver_flat_base{};

    // The flat index a (vc, channel) pair maps to, bounds-checked. The prefix sums are the whole
    // answer; these just keep every consumer from re-writing base + i without the check.
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
 * All families' arity rules live here, next to the wiring rules that produce them. Topology
 * gates the channel counts (1D has its own counts and never creates VC1/VC2 channels), but
 * num_vcs is deliberately config-only and topology-independent: a 1D router with requires_vc1
 * reports 2 VCs while creating zero VC1 channels -- that existing oddity is preserved, not
 * fixed, and consumers already tolerate counts exceeding created channels.
 *
 * The derivation emits prefix sums for every flat base and enforces the num_max_* ceilings,
 * turning the capacity comments elsewhere into guarantees at the one construction site. The
 * chip arrives as its whole per-direction capability set: this router's own capability and the
 * chip's Z-port role are both read off it, so a caller cannot spell one chip two ways, and an
 * impossible pairing of the two is unrepresentable rather than merely rejected.
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
 * The per-VC turn table of one router: which downstream routers its receiver feeds on each VC.
 * Keyed by VC alone -- a router services exactly one receiver per active VC by construction
 * (router_vc_shape gives it one), so there is no receiver dimension to index. An empty vector
 * means nothing is wired on that VC.
 */
using RouterTurnSet = std::array<std::vector<ConnectionTarget>, builder_config::MAX_NUM_VCS>;

/**
 * @brief The one turn-set derivation for every router
 *
 * Behaviour is keyed on what role the ports have, not which ports they are:
 * - A port with no routing direction (facing Z, capability INTERMESH) gets the boundary
 *   template: the full non-self set on VC1, typed from-boundary. Its VC0 senders are fed by
 *   the mesh routers' boundary targets on their own turn sets.
 * - A routing-direction port gets its turn set from the wires_into primitive: 1D is the
 *   opposite direction; non-express 2D is every non-self cardinal; express adds the express rule
 *   (an intramesh X ingress unwires from intramesh Y, a landing X ingress does not).
 * - The chip's Z port enters the set only when it has one: an express chord is an ordinary
 *   same-VC target; an intermesh boundary is reached through the boundary target on VC0 (and on
 *   VC1 only in pass-through mode); nothing exists without the port.
 *
 * The VC facts arrive as the same IntermeshVCConfig the shape derivation takes, so a caller
 * cannot spell one fabric two ways (e.g. requires_vc1 for the shape but not for the turn set),
 * and the chip arrives as the same per-direction capability set for the same reason.
 */
RouterTurnSet turn_set_for_router(
    Topology topology,
    RoutingDirection facing,
    const PerDirectionCapabilities& chip_capabilities,
    bool express_routing_enabled,
    const IntermeshVCConfig* vc_config);

/**
 * The whole wiring answer for one router archetype: its channel shape and its turn set, derived
 * together from one fact tuple. Callers that need only one side can call router_vc_shape or
 * turn_set_for_router directly; callers that need both (the per-router build, test fixtures)
 * should derive them together so the facts cannot diverge between the two.
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
 * Which datamover builder owns a router's sender channels on a VC. With the tensix (MUX)
 * extension every VC0 channel is owned by the TENSIX builder feeding the ERISC router; VC1/VC2
 * are always ERISC. `downstream_is_tensix_builder` is the one link-scope layout fact -- it can
 * differ per eth channel via is_dispatch_link in MUX mode -- so it arrives at the use site
 * rather than being baked into any archetype.
 */
enum class BuilderType : uint8_t {
    ERISC = 0,
    TENSIX = 1,
};

inline BuilderType builder_type_for_vc(uint32_t vc, bool downstream_is_tensix_builder) {
    return (vc == 0 && downstream_is_tensix_builder) ? BuilderType::TENSIX : BuilderType::ERISC;
}

}  // namespace tt::tt_fabric
