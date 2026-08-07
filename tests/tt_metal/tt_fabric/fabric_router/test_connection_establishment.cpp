// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <map>
#include <set>
#include <tuple>
#include <vector>

#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/fabric_builder_helpers.hpp"
#include "tt_metal/fabric/builder/router_wiring_rules.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

using namespace tt::tt_fabric;

/**
 * Local connection establishment, driven end to end by the turn sets.
 *
 * A host-side router carries the channel shape and the turn set derived from the SAME facts, and
 * establish() mirrors the merged production pass: every direction-matching target in the source
 * router's turn set is recorded against the router present in that direction, with the
 * destination sender slot computed from the production direction<->slot bijection -- never a
 * hand-written channel number. Absent directions are skipped, which is how edge devices fall out.
 *
 * What is pinned here:
 * - Full device: every wired turn is recorded exactly once, VC-preserving, with per-mesh and
 *   per-boundary cardinalities.
 * - Destination slots: in-bounds for the destination's shape, never aliasing another
 *   producer's slot, and never the VC0 worker slot.
 * - Edge devices: absent directions are skipped by direction matching.
 * - Establishment order does not change the resulting registry.
 * - Dimension order is visible in an express chip's registry: X-facing routers emit exactly one
 *   turn (around the X ring).
 *
 * These are host-side logic tests; no device/UMD initialization.
 */

namespace {

// The sender slot a producer facing `producer` occupies on the downstream router facing
// `downstream`, per the production placement rule.
uint32_t dest_slot(RoutingDirection producer, RoutingDirection downstream, uint32_t target_vc) {
    return builder::get_downstream_sender_channel_for_vc(
        /*is_2d_routing=*/true,
        target_vc,
        builder::routing_direction_to_eth_direction(producer),
        builder::routing_direction_to_eth_direction(downstream));
}

bool same_record(const RouterConnectionRecord& a, const RouterConnectionRecord& b) {
    return a.source_node == b.source_node && a.source_direction == b.source_direction &&
           a.source_eth_chan == b.source_eth_chan && a.source_vc == b.source_vc &&
           a.source_receiver_channel == b.source_receiver_channel && a.dest_node == b.dest_node &&
           a.dest_direction == b.dest_direction && a.dest_eth_chan == b.dest_eth_chan && a.dest_vc == b.dest_vc &&
           a.dest_sender_channel == b.dest_sender_channel;
}

std::vector<RouterConnectionRecord> select_records(
    const ConnectionRegistry& registry, bool (*pred)(const RouterConnectionRecord&)) {
    std::vector<RouterConnectionRecord> out;
    for (const auto& c : registry.get_all_connections()) {
        if (pred(c)) {
            out.push_back(c);
        }
    }
    return out;
}

bool is_zward(const RouterConnectionRecord& c) { return c.dest_direction == RoutingDirection::Z; }
bool is_fromz(const RouterConnectionRecord& c) { return c.source_direction == RoutingDirection::Z; }
bool is_cardinal(const RouterConnectionRecord& c) {
    return c.dest_direction != RoutingDirection::Z && c.source_direction != RoutingDirection::Z;
}

}  // namespace

class ConnectionEstablishmentTest : public ::testing::Test {
protected:
    // One host-side router: shape and turn set from the same facts, so the channel layout and the
    // turn table cannot disagree.
    struct HostRouter {
        FabricNodeId node;
        RoutingDirection facing;
        RouterVcShape shape;
        RouterTurnSet turns;
    };

    IntermeshVCConfig vc1_config_ = IntermeshVCConfig::full_mesh();
    ConnectionRegistry registry_;

    HostRouter make_mesh_router(RoutingDirection facing, uint32_t id, ZPortRole chip_role, bool enable_vc1) {
        const auto archetype = router_archetype(
            Topology::Mesh,
            facing,
            EdgeCapability::INTRAMESH_CARDINAL,
            chip_role,
            /*express_routing_enabled=*/false,
            enable_vc1 ? &vc1_config_ : nullptr);
        return HostRouter{FabricNodeId(MeshId{0}, id), facing, archetype.shape, archetype.turns};
    }

    HostRouter make_boundary_router(uint32_t id) {
        const auto archetype = router_archetype(
            Topology::Mesh,
            RoutingDirection::Z,
            EdgeCapability::INTERMESH,
            ZPortRole::INTERMESH_BOUNDARY,
            /*express_routing_enabled=*/false,
            &vc1_config_);
        return HostRouter{FabricNodeId(MeshId{0}, id), RoutingDirection::Z, archetype.shape, archetype.turns};
    }

    HostRouter make_express_mesh_router(RoutingDirection facing, uint32_t id) {
        const auto archetype = router_archetype(
            Topology::Torus,
            facing,
            EdgeCapability::INTRAMESH_CARDINAL,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            nullptr);
        return HostRouter{FabricNodeId(MeshId{0}, id), facing, archetype.shape, archetype.turns};
    }

    HostRouter make_chord_router(uint32_t id) {
        const auto archetype = router_archetype(
            Topology::Torus,
            RoutingDirection::Z,
            EdgeCapability::INTRAMESH_EXPRESS,
            ZPortRole::EXPRESS_CHORD,
            /*express_routing_enabled=*/true,
            nullptr);
        return HostRouter{FabricNodeId(MeshId{0}, id), RoutingDirection::Z, archetype.shape, archetype.turns};
    }

    // The merged production pass: every direction-matching target in the source turn set is recorded
    // against the router present in that direction. A target whose direction has no router is
    // skipped -- the edge-device mechanism.
    void establish(HostRouter& source, const std::map<RoutingDirection, HostRouter*>& present) {
        for (uint32_t vc = 0; vc < builder_config::MAX_NUM_VCS; ++vc) {
            for (const auto& target : source.turns[vc]) {
                TT_FATAL(target.target_direction.has_value(), "local connection target must name a direction");
                const auto dest_dir = *target.target_direction;
                if (!present.contains(dest_dir)) {
                    continue;
                }
                auto* dest = present.at(dest_dir);
                registry_.record_connection(RouterConnectionRecord{
                    .source_node = source.node,
                    .source_direction = source.facing,
                    .source_eth_chan = 0,
                    .source_vc = vc,
                    .source_receiver_channel = 0,
                    .dest_node = dest->node,
                    .dest_direction = dest->facing,
                    .dest_eth_chan = 0,
                    .dest_vc = target.target_vc,
                    .dest_sender_channel = dest_slot(source.facing, dest_dir, target.target_vc),
                });
            }
        }
    }

    void establish_all(std::vector<HostRouter>& routers) {
        std::map<RoutingDirection, HostRouter*> present;
        for (auto& r : routers) {
            present[r.facing] = &r;
        }
        for (auto& r : routers) {
            establish(r, present);
        }
    }
};

// ============================================================================
// Full device: 4 mesh routers + an intermesh boundary, VC1 enabled
// ============================================================================

TEST_F(ConnectionEstablishmentTest, FullDevice_EveryWiredTurnIsRecordedOnce) {
    std::vector<HostRouter> routers;
    for (auto facing : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        routers.push_back(make_mesh_router(facing, static_cast<uint32_t>(facing), ZPortRole::INTERMESH_BOUNDARY, true));
    }
    routers.push_back(make_boundary_router(100));

    establish_all(routers);

    // Per mesh router: VC0 wires {3 cardinals + boundary}, VC1 wires {3 cardinals} -> 7 each.
    // The boundary wires its 4-wide VC1 fanout. 4*7 + 4 = 32.
    EXPECT_EQ(registry_.size(), 32);
    EXPECT_EQ(select_records(registry_, is_cardinal).size(), 24);  // 4 routers x (3 VC0 + 3 VC1)
    EXPECT_EQ(select_records(registry_, is_zward).size(), 4);
    EXPECT_EQ(select_records(registry_, is_fromz).size(), 4);

    // No VC crossover anywhere: the boundary turn stays on VC0, the from-boundary fanout on VC1,
    // and cardinal turns never change VC.
    for (const auto& c : registry_.get_all_connections()) {
        EXPECT_EQ(c.source_vc, c.dest_vc);
    }
    for (const auto& c : select_records(registry_, is_zward)) {
        EXPECT_EQ(c.source_vc, 0);
    }
    for (const auto& c : select_records(registry_, is_fromz)) {
        EXPECT_EQ(c.source_vc, 1);
    }

    // Each mesh router turns toward the boundary exactly once; the boundary reaches every mesh
    // router exactly once in each direction.
    for (const auto& r : routers) {
        if (r.facing == RoutingDirection::Z) {
            continue;
        }
        const auto outgoing = registry_.get_connections_by_source_node(r.node);
        EXPECT_EQ(std::count_if(outgoing.begin(), outgoing.end(), is_zward), 1)
            << "mesh router " << enchantum::to_string(r.facing);
    }
    const auto z_node = routers.back().node;
    EXPECT_EQ(registry_.get_connections_by_source_node(z_node).size(), 4);
    EXPECT_EQ(registry_.get_connections_by_dest_node(z_node).size(), 4);
}

TEST_F(ConnectionEstablishmentTest, FullDevice_DestSlotsInBoundsDistinctAndWorkerSlotUntouched) {
    std::vector<HostRouter> routers;
    for (auto facing : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        routers.push_back(make_mesh_router(facing, static_cast<uint32_t>(facing), ZPortRole::INTERMESH_BOUNDARY, true));
    }
    routers.push_back(make_boundary_router(100));

    establish_all(routers);

    std::map<RoutingDirection, HostRouter*> by_facing;
    for (auto& r : routers) {
        by_facing[r.facing] = &r;
    }

    // Per destination router and VC: every producer lands in-bounds on a distinct slot, and the
    // VC0 worker slot (0) is never taken by a wired producer.
    std::map<std::pair<RoutingDirection, uint32_t>, std::set<uint32_t>> slots_per_dest_vc;
    for (const auto& c : registry_.get_all_connections()) {
        const auto* dest = by_facing.at(c.dest_direction);
        ASSERT_LT(c.dest_sender_channel, dest->shape.sender_counts[c.dest_vc])
            << "slot out of bounds on " << enchantum::to_string(c.dest_direction) << " VC" << c.dest_vc;
        if (c.dest_vc == 0) {
            EXPECT_NE(c.dest_sender_channel, 0) << "VC0 worker slot taken by a wired producer";
        }
        EXPECT_TRUE((slots_per_dest_vc[{c.dest_direction, c.dest_vc}].insert(c.dest_sender_channel).second))
            << "two producers alias slot " << c.dest_sender_channel << " on " << enchantum::to_string(c.dest_direction)
            << " VC" << c.dest_vc;
    }

    // Spot-check the shape of one mesh router's VC1 occupancy: the three cardinal producers take
    // slots 0-2 and the from-boundary producer takes slot 3 -- the slot the boundary-chip channel
    // mapping reserves beyond the legacy three.
    const auto& vc1_slots = slots_per_dest_vc[{RoutingDirection::N, 1}];
    EXPECT_EQ(vc1_slots.size(), 4);
    EXPECT_TRUE(vc1_slots.contains(3));
}

// ============================================================================
// Edge devices: absent directions are skipped by direction matching
// ============================================================================

TEST_F(ConnectionEstablishmentTest, EdgeDevice_SkipsAbsentDirections) {
    std::vector<HostRouter> routers;
    routers.push_back(make_mesh_router(RoutingDirection::N, 0, ZPortRole::INTERMESH_BOUNDARY, false));
    routers.push_back(make_mesh_router(RoutingDirection::E, 1, ZPortRole::INTERMESH_BOUNDARY, false));
    routers.push_back(make_boundary_router(100));

    establish_all(routers);

    // N wires {E, Z}, E wires {N, Z}, the boundary wires {N, E}: 2+2+2 = 6 of a full device's
    // intent. S and W are skipped by direction matching.
    EXPECT_EQ(registry_.size(), 6);
    for (const auto& c : registry_.get_all_connections()) {
        EXPECT_NE(c.dest_direction, RoutingDirection::S);
        EXPECT_NE(c.dest_direction, RoutingDirection::W);
    }
    const auto z_outgoing = registry_.get_connections_by_source_node(routers.back().node);
    EXPECT_EQ(z_outgoing.size(), 2);
}

TEST_F(ConnectionEstablishmentTest, ThreeMeshDevice_RouterCountScalesTheRegistry) {
    std::vector<HostRouter> routers;
    routers.push_back(make_mesh_router(RoutingDirection::N, 0, ZPortRole::INTERMESH_BOUNDARY, false));
    routers.push_back(make_mesh_router(RoutingDirection::E, 1, ZPortRole::INTERMESH_BOUNDARY, false));
    routers.push_back(make_mesh_router(RoutingDirection::S, 2, ZPortRole::INTERMESH_BOUNDARY, false));
    routers.push_back(make_boundary_router(100));

    establish_all(routers);

    // Each mesh router wires {2 cardinal peers + boundary} = 3; the boundary wires 3: 3*3 + 3 = 12.
    EXPECT_EQ(registry_.size(), 12);
    EXPECT_EQ(select_records(registry_, is_zward).size(), 3);
    EXPECT_EQ(select_records(registry_, is_fromz).size(), 3);
}

// ============================================================================
// Registry-level properties
// ============================================================================

TEST_F(ConnectionEstablishmentTest, NoDuplicateConnections) {
    std::vector<HostRouter> routers;
    routers.push_back(make_mesh_router(RoutingDirection::N, 0, ZPortRole::INTERMESH_BOUNDARY, false));
    routers.push_back(make_boundary_router(100));

    establish_all(routers);

    ASSERT_EQ(registry_.size(), 2);
    std::set<std::tuple<FabricNodeId, FabricNodeId, uint32_t>> unique_turns;
    for (const auto& c : registry_.get_all_connections()) {
        unique_turns.emplace(c.source_node, c.dest_node, c.source_vc);
    }
    EXPECT_EQ(unique_turns.size(), 2);
}

TEST_F(ConnectionEstablishmentTest, EstablishmentOrderDoesNotChangeTheResult) {
    auto build_registry = [this](bool boundary_first) {
        registry_.clear();
        std::vector<HostRouter> routers;
        routers.push_back(make_mesh_router(RoutingDirection::N, 0, ZPortRole::INTERMESH_BOUNDARY, false));
        routers.push_back(make_boundary_router(100));
        if (boundary_first) {
            std::reverse(routers.begin(), routers.end());
        }
        establish_all(routers);
        return registry_.get_all_connections();
    };

    const auto mesh_first = build_registry(false);
    const auto boundary_first = build_registry(true);

    ASSERT_EQ(mesh_first.size(), boundary_first.size());
    for (const auto& record : mesh_first) {
        EXPECT_TRUE(std::any_of(boundary_first.begin(), boundary_first.end(), [&](const auto& other) {
            return same_record(record, other);
        })) << "a connection present mesh-first is missing boundary-first";
    }
}

// ============================================================================
// Express chip: dimension order is visible in the registry
// ============================================================================

TEST_F(ConnectionEstablishmentTest, ExpressChip_XRoutersEmitOnlyAroundTheXRing) {
    std::vector<HostRouter> routers;
    for (auto facing : {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}) {
        routers.push_back(make_express_mesh_router(facing, static_cast<uint32_t>(facing)));
    }
    routers.push_back(make_chord_router(100));

    establish_all(routers);

    // N/S wire {opposite Y, E, W, chord} = 4 each; E/W wire only the opposite X = 1 each; the
    // chord wires all four cardinals. 2*4 + 2*1 + 4 = 14.
    EXPECT_EQ(registry_.size(), 14);
    for (const auto& c : registry_.get_all_connections()) {
        const bool x_source = c.source_direction == RoutingDirection::E || c.source_direction == RoutingDirection::W;
        if (x_source) {
            const bool x_dest = c.dest_direction == RoutingDirection::E || c.dest_direction == RoutingDirection::W;
            EXPECT_TRUE(x_dest) << "an intramesh X producer was wired back into Y: "
                                << enchantum::to_string(c.dest_direction);
        }
    }
    // Every turn lands in-bounds on the destination express router's five-wide VC0.
    std::map<RoutingDirection, HostRouter*> by_facing;
    for (auto& r : routers) {
        by_facing[r.facing] = &r;
    }
    for (const auto& c : registry_.get_all_connections()) {
        EXPECT_LT(c.dest_sender_channel, by_facing.at(c.dest_direction)->shape.sender_counts[0]);
    }
}
