// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <numeric>
#include <optional>
#include <vector>

#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include "tt_metal/fabric/builder/fabric_edge_capability.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

// Forward declaration
struct IntermeshVCConfig;

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
 * - The Z-facing intermesh boundary's from-boundary VC1 fanout to every mesh direction
 * - Boundary turns to a chip's intermesh edge (any capability)
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
     * @brief The turn-matrix primitive: does the router facing `producer_direction` wire into the
     * router facing `egress_direction` on the same chip?
     *
     * The one place the wiring rule lives. Turn sets, producer arity, and the injection-flag
     * derivation are all built from this single relation, so the connection map and the guard
     * derivation cannot drift apart -- there is no set form of the rule for them to disagree with.
     *
     * The rule, per case:
     * - No U-turn: a router never wires back over its own link.
     * - Legacy (non-express): every non-self cardinal direction wires in. The extra port exists in
     *   the set only as the boundary template, when the chip's extra port is INTERMESH_BOUNDARY.
     * - Express: a Z-facing producer fans out to every non-self direction; an intramesh X producer
     *   may only continue around the X ring (dimension order); any other producer wires into every
     *   non-self direction; and the extra port exists in the set only when the chip has one
     *   (chord or boundary).
     *
     * The VC matters exactly once: for a boundary producer (a Z-facing router whose edge is
     * INTERMESH). Its VC1 receiver fans out to every non-self VC1 sender (wired), while its VC0
     * receiver crosses over onto downstream VC1 senders and feeds nothing on VC0 (not wired). So
     * the boundary-producer arm answers `vc == 1`. Every other producer is VC-agnostic on this
     * question, and for_router may pass any VC when building a turn set, since that arm is
     * unreachable there (the boundary path early-returns before consulting this primitive).
     *
     * Guard classification follows from the answers: a cardinal producer into a boundary egress is
     * NON_RING (the egress is not a protected ring), and a boundary producer into a protected
     * cardinal egress on VC1 is ENTER -- the correct landing acquisition. The first arm is not
     * "NON_RING either way": only the second depends on the VC.
     */
    static bool wires_into(
        RoutingDirection producer_direction,
        EdgeCapability producer_capability,
        RoutingDirection egress_direction,
        ZPortRole z_role,
        bool express_routing_enabled,
        uint32_t vc);

    /**
     * Per-direction capability set of one chip: each direction's edge capability, indexed by
     * RoutingDirection enum value (E=0, W=1, N=2, S=3, Z=4); nullopt where the direction is absent.
     */
    using PerDirectionCapabilities = std::array<std::optional<EdgeCapability>, 5>;

    /**
     * @brief The canonical express-endpoint chip: every cardinal intramesh, Z is the chord
     *
     * This is the capability set the express family-max count is evaluated against. It attains
     * the structural ceiling: every Y and X producer wires into an E/W facing under any capability
     * assignment, so no per-chip set produces a wider router.
     */
    static PerDirectionCapabilities canonical_express_endpoint_capabilities();

    /**
     * @brief VC0 sender slots a router facing `direction` needs on a chip with these capabilities
     *
     * The local worker plus the producers the connection map wires into it. Arity depends on
     * facing: an E/W-facing router is fed by every Y producer (the Y->X turn is legal), while
     * dimension order leaves N/S/Z-facing routers with only their Y producers. INTERMESH
     * (landing-capable) directions widen their non-self set, so per-chip callers must pass the
     * actual chip's capabilities; the canonical endpoint set is only the family-max input.
     */
    static uint32_t express_vc0_producer_arity(RoutingDirection direction, const PerDirectionCapabilities& caps);

    /**
     * @brief Uniform VC0/VC1 sender counts for the express mesh family
     *
     * The family max over facing directions of wired-producer arity on the canonical endpoint
     * chip: one flat index space per family, with per-router wiring filling a subset and
     * per-direction channel trimming as the separate L1 lever for narrowing (which evaluates
     * arity against the actual per-chip capability set, not the canonical one).
     */
    static uint32_t express_mesh_vc0_sender_count();
    static uint32_t express_mesh_vc1_sender_count();

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
    };

    /**
     * @brief The one per-VC shape derivation for any router
     *
     * All families' arity rules live here, next to the wiring rules that produce them. Topology
     * gates the channel counts (1D has its own counts and never creates VC1/VC2 channels), but
     * num_vcs is deliberately config-only and topology-independent: a 1D router with requires_vc1
     * reports 2 VCs while creating zero VC1 channels -- that existing oddity is preserved, not
     * fixed, and get_all_sender_mappings() already tolerates it.
     *
     * The derivation emits prefix sums for every flat base and enforces the num_max_* ceilings,
     * turning the capacity comments elsewhere into guarantees at the one construction site. The
     * chip's extra port arrives as its ZPortRole (boundary, chord, or none) -- the same fact the
     * connection map reads, with one spelling.
     */
    static RouterVcShape router_vc_shape(
        Topology topology,
        RoutingDirection facing,
        EdgeCapability edge_capability,
        ZPortRole z_role,
        bool express_routing_enabled,
        const IntermeshVCConfig* vc_config);

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
    // Per-VC sender arity of one 2D mesh-like router, by family: the local worker (VC0 only) plus
    // the producers wired into it on that VC. Called only by router_vc_shape; the answers are
    // read off the shape, never re-derived at a consumption site. The Z-facing intermesh boundary
    // family is separate and has its own derived accessors in builder_config
    // (num_sender_channels_intermesh_z_boundary_*).
    static uint32_t mesh_router_vc0_sender_count(ZPortRole z_role, bool express_routing_enabled);
    static uint32_t mesh_router_vc1_sender_count(ZPortRole z_role, bool express_routing_enabled);

    // The chip-level cross-check both factories run first: a Z-facing router's own edge capability
    // and the chip's extra-port role are two spellings of one fact and must agree -- a Z-facing
    // intermesh edge means role INTERMESH_BOUNDARY, a same-mesh Z edge (an express chord) means
    // role EXPRESS_CHORD, and express capability never sits on a cardinal facing. Anything else is
    // an impossible chip, which the independent parameters would otherwise make representable
    // again.
    static void validate_facing_role_consistency(
        RoutingDirection facing, EdgeCapability edge_capability, ZPortRole z_role);

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
