// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression for the producer effect derivation that selects each sender's flow-control guard.
//
// The ladder is driven from the real derived ring topologies -- the same AxisRouteTopology state
// the ControlPlane serves -- rather than hand-written predicate answers, so a disagreement between
// the derivation and the ladder shows up here. The three ring queries are re-implemented over the
// machine-free topology pair, mirroring ControlPlane::is_protected_ring_edge /
// are_same_directed_ring_edges / continuation_allowed line for line; the ControlPlane versions
// themselves are pinned CP-backed in test_express_ring_topology.cpp. Keeping this file machine-free
// means the builder ladder runs in every environment.

#include <gtest/gtest.h>

#include <enchantum/enchantum.hpp>
#include <filesystem>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/mesh_graph.hpp>

#include "cluster.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/injection_policy.hpp"
#include "tt_metal/fabric/builder/protected_domain_effect.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/axis_route_topology.hpp"

namespace tt::tt_fabric {
namespace {

constexpr auto k_cardinal = EdgeCapability::INTRAMESH_CARDINAL;
constexpr auto k_express = EdgeCapability::INTRAMESH_EXPRESS;
constexpr auto k_intermesh = EdgeCapability::INTERMESH;

// The validated four-Galaxy fixture: ex4 and ex8 chords over a 32-row column with the cardinal end
// wrap present and the X ring closed -- express_links_32x4_mesh_graph_descriptor.textproto.
struct QuadGalaxy {
    MeshGraph graph;
    AxisRouteTopology y_rings;  // the express axis (dim 0)
    AxisRouteTopology x_rings;  // the ordinary X ring (dim 1)

    QuadGalaxy() :
        graph(tt::tt_metal::ClusterType::BLACKHOLE_GALAXY, descriptor_path()),
        y_rings(derive_express_ring_topology(graph, MeshId{0}).value()),
        x_rings(derive_ordinary_ring_topology(graph, MeshId{0}, /*axis=*/1).value()) {}

    static std::string descriptor_path() {
        return (std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
                "tests/tt_metal/tt_fabric/custom_mesh_descriptors" /
                "express_links_32x4_mesh_graph_descriptor.textproto")
            .string();
    }

    // The column-0 node on logical row `row`.
    FabricNodeId node_at(uint32_t row) const { return FabricNodeId{MeshId{0}, static_cast<std::uint32_t>(row * 4)}; }

    // --- The three ring queries, mirroring the ControlPlane methods over the machine-free state ---

    const AxisRouteTopology* ring_for_direction(RoutingDirection direction) const {
        const bool orthogonal = direction == RoutingDirection::E || direction == RoutingDirection::W;
        return orthogonal ? &x_rings : &y_rings;
    }

    int coord_of(FabricNodeId node, int axis_dim) const {
        return static_cast<int>(graph.chip_to_coordinate(node.mesh_id, node.chip_id)[axis_dim]);
    }

    std::optional<int> neighbor_coord(FabricNodeId node, RoutingDirection direction, int axis_dim) const {
        const auto& connectivity = graph.get_intra_mesh_connectivity();
        for (const auto& [neighbor_chip, edge] : connectivity[*node.mesh_id][node.chip_id]) {
            if (edge.port_direction == direction) {
                return static_cast<int>(graph.chip_to_coordinate(node.mesh_id, neighbor_chip)[axis_dim]);
            }
        }
        return std::nullopt;
    }

    bool is_protected_ring_edge(FabricNodeId local, RoutingDirection egress) const {
        const auto* rings = ring_for_direction(egress);
        if (rings == nullptr) {
            return false;
        }
        const int coord = coord_of(local, rings->axis_dim);
        const auto peer = neighbor_coord(local, egress, rings->axis_dim);
        if (!peer.has_value() || rings->is_leaf(coord) || rings->is_leaf(*peer)) {
            return false;
        }
        if (rings->domain_of[coord] != rings->domain_of[*peer]) {
            return false;  // a crossover joins two rings; it belongs to neither
        }
        return rings->ring_distance(rings->domain_of[coord], coord, *peer) == 1;
    }

    bool are_same_directed_ring_edges(FabricNodeId local, RoutingDirection ingress, RoutingDirection egress) const {
        const auto* rings = ring_for_direction(ingress);
        // Both hops must ride the same ring, so a turn between the two axes is never the same view.
        if (rings == nullptr || rings != ring_for_direction(egress) || !is_protected_ring_edge(local, ingress) ||
            !is_protected_ring_edge(local, egress)) {
            return false;
        }
        const int coord = coord_of(local, rings->axis_dim);
        const auto from = neighbor_coord(local, ingress, rings->axis_dim);
        const auto to = neighbor_coord(local, egress, rings->axis_dim);
        if (!from.has_value() || !to.has_value()) {
            return false;
        }
        const int domain = rings->domain_of[coord];
        const int n = static_cast<int>(rings->forward_cycle[domain].size());
        const auto forward_step = [&](int a, int b) {
            return (rings->pos_in_domain[b] - rings->pos_in_domain[a] + n) % n == 1;
        };
        // Arriving from the ingress neighbor and leaving toward the egress one must step the same
        // way around the cycle: both forward, or both reverse.
        return (forward_step(*from, coord) && forward_step(coord, *to)) ||
               (forward_step(coord, *from) && forward_step(*to, coord));
    }

    bool continuation_allowed(FabricNodeId local, RoutingDirection ingress, RoutingDirection egress) const {
        const auto* rings = &y_rings;
        if (!this->is_protected_ring_edge(local, egress)) {
            return true;  // nothing protected is being acquired, so nothing to gate
        }
        const int row = coord_of(local, rings->axis_dim);
        const auto from_row = neighbor_coord(local, ingress, rings->axis_dim);
        if (!from_row.has_value() || rings->is_leaf(row) || rings->is_leaf(*from_row)) {
            return true;  // worker injection, or arrival over a leaf-run or anchor edge
        }
        if (rings->domain_of[*from_row] == rings->domain_of[row]) {
            return true;  // already riding this ring: transit, not an acquisition
        }
        // Arrived over a crossover, so the packet carries the ingress side's family. Only the family
        // marked as continuing may acquire the other; the reverse crossing is terminal.
        return rings->domain_of[*from_row] == rings->continue_src_domain;
    }
};

ProtectedRingQueries bind(const QuadGalaxy& fixture, uint32_t row) {
    const FabricNodeId node = fixture.node_at(row);
    ProtectedRingQueries queries;
    queries.is_protected_ring_edge = [&fixture, node](RoutingDirection egress) {
        return fixture.is_protected_ring_edge(node, egress);
    };
    queries.are_same_directed_ring_edges = [&fixture, node](RoutingDirection ingress, RoutingDirection egress) {
        return fixture.are_same_directed_ring_edges(node, ingress, egress);
    };
    queries.continuation_allowed = [&fixture, node](RoutingDirection ingress, RoutingDirection egress) {
        return fixture.continuation_allowed(node, ingress, egress);
    };
    return queries;
}

// --- Node Y=2: an ex4 express node ---

TEST(ProtectedDomainEffectsTest, Row2ExpressEgressCarriesBothRoles) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);

    // This is the case the whole derivation exists for: one express output, two producers, two roles.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::Z), ProtectedDomainEffect::ENTER);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::N, k_cardinal, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::REMAIN);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::S, k_cardinal, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::ENTER);

    // Only the acquisition becomes an injection channel.
    EXPECT_FALSE(is_injection_effect(ProtectedDomainEffect::REMAIN));
    EXPECT_TRUE(is_injection_effect(ProtectedDomainEffect::ENTER));
}

TEST(ProtectedDomainEffectsTest, Row2ReverseCardinalEgressIsSymmetric) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);

    // e(2->1) belongs to the reverse orientation: express-face transit remains, leaf attachment enters.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::Z, k_express, RoutingDirection::N, k_cardinal),
        ProtectedDomainEffect::REMAIN);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::S, k_cardinal, RoutingDirection::N, k_cardinal),
        ProtectedDomainEffect::ENTER);
}

TEST(ProtectedDomainEffectsTest, Row2TurnOntoXAcquiresTheXRing) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);

    // Y->X is an X acquisition regardless of which Y producer feeds it, including the express one.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::E), ProtectedDomainEffect::ENTER);
    for (const auto ingress : {RoutingDirection::N, RoutingDirection::S, RoutingDirection::Z}) {
        const auto capability = ingress == RoutingDirection::Z ? k_express : k_cardinal;
        EXPECT_EQ(
            classify_producer_effect(q, ingress, capability, RoutingDirection::E, k_cardinal),
            ProtectedDomainEffect::ENTER);
    }
    // X transit stays transit.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::W, k_cardinal, RoutingDirection::E, k_cardinal),
        ProtectedDomainEffect::REMAIN);
}

// --- Node Y=3: a leaf ---

TEST(ProtectedDomainEffectsTest, LeafCardinalEgressIsNotRingAcquisition) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 3);

    // Cardinal moves to the anchor or the paired leaf are attachments, not ex4 acquisitions.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::N), ProtectedDomainEffect::NON_RING);
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::S), ProtectedDomainEffect::NON_RING);
}

TEST(ProtectedDomainEffectsTest, LeafStillNeedsXRingGuards) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 3);

    // A Y leaf is still on the X ring, so flow control can never be decided per chip.
    EXPECT_EQ(classify_worker_effect(q, RoutingDirection::E), ProtectedDomainEffect::ENTER);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::N, k_cardinal, RoutingDirection::E, k_cardinal),
        ProtectedDomainEffect::ENTER);
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::W, k_cardinal, RoutingDirection::E, k_cardinal),
        ProtectedDomainEffect::REMAIN);
}

// --- Cross-family turns ---

TEST(ProtectedDomainEffectsTest, ContinueCrossoverEntersButLandOnlyIsNonCanonical) {
    const QuadGalaxy fixture;

    // CONTINUE: 0 (ex8) -> 1 (land) -> 2 (first ex4 cyclic edge). The hop 0->1 arrives on row 1's
    // N-facing port and the egress toward 2 is S.
    const auto at_row1 = bind(fixture, 1);
    EXPECT_EQ(
        classify_producer_effect(at_row1, RoutingDirection::N, k_cardinal, RoutingDirection::S, k_cardinal),
        ProtectedDomainEffect::ENTER);

    // LAND_ONLY: 6 (ex4) -> 7 (land) -> 8 (first ex8 cyclic edge) is terminal in Y, so the turn is
    // outside the canonical route set even though it remains physically wireable.
    const auto at_row7 = bind(fixture, 7);
    EXPECT_EQ(
        classify_producer_effect(at_row7, RoutingDirection::N, k_cardinal, RoutingDirection::S, k_cardinal),
        ProtectedDomainEffect::NON_CANONICAL);
    EXPECT_FALSE(is_injection_effect(ProtectedDomainEffect::NON_CANONICAL));
}

// --- Intermesh landing and dimension order ---

TEST(ProtectedDomainEffectsTest, IntermeshLandingAcquiresItsFirstProtectedEgress) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);

    // A landed carrier holds no position on this mesh's rings, so its first protected output is an
    // acquisition -- even where an equivalent intramesh producer would have been transit.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::N, k_intermesh, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::ENTER);
    // And the landing is exempt from dimension order, so an E-facing boundary port may begin Y.
    EXPECT_EQ(
        classify_producer_effect(q, RoutingDirection::E, k_intermesh, RoutingDirection::Z, k_express),
        ProtectedDomainEffect::ENTER);
}

TEST(ProtectedDomainEffectsTest, IntrameshXIntoYIsRejectedRatherThanClassified) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);

    // Connection mapping unwires this producer, so reaching the derivation means the maps and this
    // ladder disagree. It fails rather than returning a guess.
    EXPECT_ANY_THROW(classify_producer_effect(q, RoutingDirection::E, k_cardinal, RoutingDirection::Z, k_express));
    EXPECT_TRUE(is_static_dor_forbidden(RoutingDirection::E, k_cardinal, RoutingDirection::Z, k_express));
    EXPECT_FALSE(is_static_dor_forbidden(RoutingDirection::E, k_intermesh, RoutingDirection::Z, k_express));
}

// --- What the replaced heuristic got wrong ---

TEST(ProtectedDomainEffectsTest, OneAxisPairYieldsTwoDifferentGuards) {
    // Why no axis-based rule can work. Both producers below are a Y ingress feeding a Y egress on the
    // same node and the same output, so any rule keyed on the axis pair must return one answer for
    // both -- yet one is transit and the other an acquisition. The replaced heuristic collapsed them
    // and gave the leaf-fed acquisition the weaker guard, which is the flow-control violation this
    // derivation removes.
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);

    const auto from_ring = classify_producer_effect(q, RoutingDirection::N, k_cardinal, RoutingDirection::Z, k_express);
    const auto from_leaf = classify_producer_effect(q, RoutingDirection::S, k_cardinal, RoutingDirection::Z, k_express);

    EXPECT_NE(from_ring, from_leaf);
    EXPECT_FALSE(is_injection_effect(from_ring));
    EXPECT_TRUE(is_injection_effect(from_leaf));
}

TEST(ProtectedDomainEffectsTest, EffectNamesAreStable) {
    EXPECT_EQ(enchantum::to_string(ProtectedDomainEffect::NON_RING), "NON_RING");
    EXPECT_EQ(enchantum::to_string(ProtectedDomainEffect::REMAIN), "REMAIN");
    EXPECT_EQ(enchantum::to_string(ProtectedDomainEffect::ENTER), "ENTER");
    EXPECT_EQ(enchantum::to_string(ProtectedDomainEffect::NON_CANONICAL), "NON_CANONICAL");
}

// --- The slot-level derivation: bound facts to per-channel flags ---
//
// ExpressInjectionPolicy takes only bound facts (the ring predicates, the chip's capability set,
// its Z port role), so the unified slot walk is drivable from the derived topologies without a
// ControlPlane. The expected effects are asserted above; these cases add the slot arithmetic around
// them: which producer lands on which channel, the VC1 shift, the dimension-order skip, and the
// absent-direction skip.

TEST(ProtectedDomainEffectsTest, ExpressEgressFlagsLandOnTheirProducerSlots) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);
    // Node Y=2's capability set: every cardinal intramesh, Z is the chord.
    const auto caps = canonical_express_endpoint_capabilities();

    // Z egress, VC0. Slots [worker, E, W, N, S]: the worker and the leaf-fed S producer acquire;
    // the ring-fed N producer is transit; both X producers are dimension-order-unwired.
    const builder::RouterProducerSlots slots(
        builder::routing_direction_to_eth_direction(RoutingDirection::Z), {5, 4, 0});
    const ExpressInjectionPolicy policy(q, caps, ZPortRole::EXPRESS_CHORD, RoutingDirection::Z, k_express);
    const auto flags = compute_sender_channel_injection_flags(slots, /*vc=*/0, policy);
    EXPECT_EQ(flags, std::vector<bool>({true, false, false, false, true}));
}

TEST(ProtectedDomainEffectsTest, UnprotectedEgressFlagsNothing) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);
    const auto caps = canonical_express_endpoint_capabilities();

    // S egress toward the leaf: 2->3 is not a protected edge, so every wired producer is NON_RING
    // -- including the chord producer, which wires in but acquires nothing.
    const builder::RouterProducerSlots slots(
        builder::routing_direction_to_eth_direction(RoutingDirection::S), {5, 4, 0});
    const ExpressInjectionPolicy policy(q, caps, ZPortRole::EXPRESS_CHORD, RoutingDirection::S, k_cardinal);
    const auto flags = compute_sender_channel_injection_flags(slots, /*vc=*/0, policy);
    EXPECT_EQ(flags, std::vector<bool>({false, false, false, false, false}));
}

TEST(ProtectedDomainEffectsTest, ReverseCardinalEgressFlagsTheLeafAndWorker) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);
    const auto caps = canonical_express_endpoint_capabilities();

    // N egress (2->1, on the reverse ring). Slots [worker, E, W, S, Z]: the worker and the
    // leaf-fed S producer acquire; the chord producer is reverse-ring transit; X unwired.
    const builder::RouterProducerSlots slots(
        builder::routing_direction_to_eth_direction(RoutingDirection::N), {5, 4, 0});
    const ExpressInjectionPolicy policy(q, caps, ZPortRole::EXPRESS_CHORD, RoutingDirection::N, k_cardinal);
    const auto flags = compute_sender_channel_injection_flags(slots, /*vc=*/0, policy);
    EXPECT_EQ(flags, std::vector<bool>({true, false, false, true, false}));
}

TEST(ProtectedDomainEffectsTest, LeafRouterSkipsTheAbsentZSlot) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 3);
    // A leaf terminates no chord: no Z entry, so the family-max slot for the Z producer stays
    // unfilled -- per-router wiring fills a subset of the family count.
    auto caps = canonical_express_endpoint_capabilities();
    caps.at(RoutingDirection::Z) = std::nullopt;

    // N egress toward the anchor: an attachment, not a ring acquisition -- the acquisition lives
    // at the anchor's Z sender (the first case above), not at the leaf's own worker.
    //
    // The all-false vector does not itself discriminate the absent-direction skip: every slot is
    // false for its own reason (unprotected egress, dimension order), and a leaf's Z slot can
    // never legitimately flag. What the case guards is the guard itself: dropping the nullopt
    // check dereferences it, which crashes here rather than failing cleanly.
    const builder::RouterProducerSlots slots(
        builder::routing_direction_to_eth_direction(RoutingDirection::N), {5, 4, 0});
    const ExpressInjectionPolicy policy(q, caps, ZPortRole::NONE, RoutingDirection::N, k_cardinal);
    const auto flags = compute_sender_channel_injection_flags(slots, /*vc=*/0, policy);
    EXPECT_EQ(flags, std::vector<bool>({false, false, false, false, false}));
}

TEST(ProtectedDomainEffectsTest, Vc1ShiftsProducerSlotsAndLandedCarrierAcquires) {
    const QuadGalaxy fixture;
    const auto q = bind(fixture, 2);
    // The E port is an intermesh landing: exempt from dimension order, and its first protected
    // egress is an acquisition on the landed VC.
    auto caps = canonical_express_endpoint_capabilities();
    caps.at(RoutingDirection::E) = k_intermesh;

    // VC1 has no worker channel, so the producer slots shift down one: [E, W, N, S] at channels
    // 0..3. The landing acquires; W is dimension-order-unwired; N is transit; S acquires.
    const builder::RouterProducerSlots slots(
        builder::routing_direction_to_eth_direction(RoutingDirection::Z), {5, 4, 0});
    const ExpressInjectionPolicy policy(q, caps, ZPortRole::EXPRESS_CHORD, RoutingDirection::Z, k_express);
    const auto flags = compute_sender_channel_injection_flags(slots, /*vc=*/1, policy);
    EXPECT_EQ(flags, std::vector<bool>({true, false, false, true}));
}

}  // namespace
}  // namespace tt::tt_fabric
