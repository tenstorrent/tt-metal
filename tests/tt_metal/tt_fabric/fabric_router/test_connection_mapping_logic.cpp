// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <set>
#include <tuple>
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

using namespace tt::tt_fabric;

/**
 * RouterConnectionMapping Tests
 *
 * These tests validate that:
 * 1. Connection mapping drives connection establishment
 * 2. configure_local_connections() handles Z↔mesh connections correctly
 * 3. Asymmetric connections (MESH_TO_Z vs Z_TO_MESH) work properly
 * 4. Edge cases (2-3 mesh routers) are handled gracefully
 *
 * Test Coverage Summary:
 * ┌────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                │ Test Name                                  │ Focus        │
 * ├─────────────────────────┼────────────────────────────────────────────┼──────────────┤
 * │ Mapping-Driven Logic    │ MappingDriven_INTRA_MESH_Connections       │ Mesh VC0     │
 * │                         │ MappingDriven_MESH_TO_Z_Connection         │ Mesh→Z       │
 * │                         │ MappingDriven_Z_TO_MESH_Connections        │ Z→Mesh       │
 * ├─────────────────────────┼────────────────────────────────────────────┼──────────────┤
 * │ Asymmetric Connections  │ AsymmetricConnections_MeshToZ_vs_ZToMesh   │ Bidirectional│
 * │                         │ AsymmetricConnections_VCAssignments        │ VC routing   │
 * ├─────────────────────────┼────────────────────────────────────────────┼──────────────┤
 * │ Edge Device Scenarios   │ EdgeDevice_TwoMeshRouters_ZMapping         │ 2-router     │
 * │                         │ EdgeDevice_ThreeMeshRouters_MappingIntent  │ 3-router     │
 * ├─────────────────────────┼────────────────────────────────────────────┼──────────────┤
 * │ Connection Filtering    │ ConnectionTypeFiltering_INTRA_MESH_Only    │ Type counts  │
 * │                         │ ConnectionTypeFiltering_LocalOnly          │ Z isolation  │
 * ├─────────────────────────┼────────────────────────────────────────────┼──────────────┤
 * │ Multi-VC Scenarios      │ MultiVC_MeshRouter_VC0_and_VC1             │ Mesh VCs     │
 * │                         │ MultiVC_ZRouter_VC0_and_VC1                │ Z VCs        │
 * ├─────────────────────────┼────────────────────────────────────────────┼──────────────┤
 * │ Direction-Based Routing │ DirectionBased_ZRouter_ChannelToDirection  │ Z channels   │
 * │                         │ DirectionBased_MeshRouter_MeshToZ          │ Z direction  │
 * ├─────────────────────────┼────────────────────────────────────────────┼──────────────┤
 * │ Regression Tests        │ Regression_MeshToMesh_StillWorks           │ Basic mesh   │
 * │                         │ Regression_1D_Topology_StillWorks          │ 1D topology  │
 * └─────────────────────────┴────────────────────────────────────────────┴──────────────┘
 *
 * Total: 15 tests across 7 categories
 */

class RouterConnectionMappingTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_shared<ConnectionRegistry>();
    }

    std::shared_ptr<ConnectionRegistry> registry_;
};

// ============================================================================
// Test 1: Mapping-Driven Connection Logic
// ============================================================================

TEST_F(RouterConnectionMappingTest, MappingDriven_INTRA_MESH_Connections) {
    // Verify that INTRA_MESH connections are established based on connection mapping
    // This is a conceptual test - actual connection establishment requires builders

    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        false);  // No Z router

    // 2D Mesh router: receiver channel 0 has 3 INTRA_MESH targets (peers in 3 directions)
    auto targets = mesh_mapping.get_downstream_targets(0, 0);
    EXPECT_EQ(targets.size(), builder_config::num_downstream_edms_2d_vc0);

    // All targets should be INTRA_MESH
    for (const auto& target : targets) {
        EXPECT_EQ(target.type, ConnectionType::INTRA_MESH);
    }
}

TEST_F(RouterConnectionMappingTest, MappingDriven_MESH_TO_Z_Connection) {
    // Verify MESH_TO_Z connection is in the mapping
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);  // Has Z router

    // Receiver channel 0 should have MESH_TO_Z target among its targets
    auto targets = mesh_mapping.get_downstream_targets(0, 0);

    auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, targets.end());
    EXPECT_EQ(mesh_to_z_it->target_vc, 0);  // Mesh VC0 → Z VC0
    ASSERT_TRUE(mesh_to_z_it->target_direction.has_value());
    EXPECT_EQ(mesh_to_z_it->target_direction.value(), RoutingDirection::Z);
}

TEST_F(RouterConnectionMappingTest, MappingDriven_Z_TO_MESH_Connections) {
    // Verify Z_TO_MESH connections are in the mapping
    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Z router VC1 receiver channel 0 should have Z_TO_MESH targets for all mesh directions (N/E/S/W intent)
    std::vector<RoutingDirection> expected_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    auto targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), builder_config::num_mesh_directions_2d);

    // Verify all expected directions are present
    for (const auto& expected_dir : expected_directions) {
        auto it = std::find_if(targets.begin(), targets.end(), [expected_dir](const ConnectionTarget& t) {
            return t.target_direction.has_value() && t.target_direction.value() == expected_dir;
        });
        ASSERT_NE(it, targets.end()) << "Missing direction: " << static_cast<int>(expected_dir);
        EXPECT_EQ(it->type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(it->target_vc, 1);  // Z VC1 → Mesh VC1
    }
}

// ============================================================================
// Test 2: Asymmetric Connection Validation
// ============================================================================

TEST_F(RouterConnectionMappingTest, AsymmetricConnections_MeshToZ_vs_ZToMesh) {
    // Verify that MESH_TO_Z and Z_TO_MESH are properly asymmetric

    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);

    auto z_mapping = RouterConnectionMapping::for_z_router();

    // MESH_TO_Z: Mesh VC0 receiver channel 0 has MESH_TO_Z target → Z VC0
    auto mesh_targets = mesh_mapping.get_downstream_targets(0, 0);
    auto mesh_to_z_it = std::find_if(mesh_targets.begin(), mesh_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, mesh_targets.end());
    EXPECT_EQ(mesh_to_z_it->target_vc, 0);

    // Z_TO_MESH: Z VC1 receiver channel 0 has 4 targets (all mesh directions) → Mesh VC1
    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), builder_config::num_mesh_directions_2d);

    for (const auto& target : z_targets) {
        EXPECT_EQ(target.type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(target.target_vc, 1);  // Different VC!
    }

    // Key insight: Different VCs, different directions, different connection types
}

TEST_F(RouterConnectionMappingTest, AsymmetricConnections_VCAssignments) {
    // Validate VC assignments are correct for asymmetric connections

    // Mesh VC0 → Z VC0 (MESH_TO_Z)
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::E,
        true);

    auto mesh_targets = mesh_mapping.get_downstream_targets(0, 0);
    auto mesh_to_z_it = std::find_if(mesh_targets.begin(), mesh_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, mesh_targets.end());
    EXPECT_EQ(mesh_to_z_it->target_vc, 0);

    // Z VC1 → Mesh VC1 (distribution)
    auto z_mapping = RouterConnectionMapping::for_z_router();

    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 4);  // 4 targets (N/E/S/W)
    for (const auto& target : z_targets) {
        EXPECT_EQ(target.target_vc, 1);
    }
}

// ============================================================================
// Test 3: Edge Device Scenarios (2-3 Mesh Routers)
// ============================================================================

TEST_F(RouterConnectionMappingTest, EdgeDevice_TwoMeshRouters_ZMapping) {
    // Z router mapping specifies intent for all mesh directions
    // configure_local_connections() will skip non-existent routers

    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Receiver channel 0 on VC1 has all mesh direction targets (intent)
    auto targets = z_mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(targets.size(), builder_config::num_mesh_directions_2d);

    for (const auto& target : targets) {
        EXPECT_EQ(target.type, ConnectionType::Z_TO_MESH);
    }

    // FabricBuilder will check if routers exist before connecting
    // This test validates the mapping is correct regardless of actual router count
}

TEST_F(RouterConnectionMappingTest, EdgeDevice_ThreeMeshRouters_MappingIntent) {
    // Validate that Z router mapping is consistent regardless of device position

    auto z_mapping_center = RouterConnectionMapping::for_z_router();

    // Receiver channel 0 on VC1 should have all mesh direction targets (N/E/S/W)
    EXPECT_TRUE(z_mapping_center.has_targets(1, 0));
    auto targets = z_mapping_center.get_downstream_targets(1, 0);
    EXPECT_EQ(targets.size(), builder_config::num_mesh_directions_2d);

    // configure_local_connections() will gracefully skip missing routers
}

// ============================================================================
// Test 4: Connection Type Filtering
// ============================================================================

TEST_F(RouterConnectionMappingTest, ConnectionTypeFiltering_INTRA_MESH_Only) {
    // configure_connection() should only handle INTRA_MESH
    // configure_local_connections() should only handle MESH_TO_Z and Z_TO_MESH

    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);

    // Count connection types
    int intra_mesh_count = 0;
    int mesh_to_z_count = 0;

    // Check all VC0 sender channels (mesh directions + Z)
    for (uint32_t ch = 0; ch < builder_config::num_sender_channels_2d_mesh + 1; ++ch) {
        auto targets = mesh_mapping.get_downstream_targets(0, ch);
        for (const auto& target : targets) {
            if (target.type == ConnectionType::INTRA_MESH) {
                intra_mesh_count++;
            } else if (target.type == ConnectionType::MESH_TO_Z) {
                mesh_to_z_count++;
            }
        }
    }

    EXPECT_GT(intra_mesh_count, 0);  // Has INTRA_MESH connections
    EXPECT_EQ(mesh_to_z_count, 1);   // Has exactly 1 MESH_TO_Z connection
}

TEST_F(RouterConnectionMappingTest, ConnectionTypeFiltering_LocalOnly) {
    // Z router should only have local connections (Z_TO_MESH)
    // No INTRA_MESH connections

    auto z_mapping = RouterConnectionMapping::for_z_router();

    int z_to_mesh_count = 0;
    int intra_mesh_count = 0;

    // Check VC0 (should be empty or reserved). Z-facing boundary VC0: worker + 4 non-self = 5.
    for (uint32_t ch = 0; ch < 5; ++ch) {
        auto vc0_targets = z_mapping.get_downstream_targets(0, ch);
        EXPECT_EQ(vc0_targets.size(), 0);  // VC0 unused for Z router
    }

    // Check VC1 (should have Z_TO_MESH for all mesh directions). From-Z fanout = 4 senders.
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto vc1_targets = z_mapping.get_downstream_targets(1, ch);
        for (const auto& target : vc1_targets) {
            if (target.type == ConnectionType::Z_TO_MESH) {
                z_to_mesh_count++;
            } else if (target.type == ConnectionType::INTRA_MESH) {
                intra_mesh_count++;
            }
        }
    }

    EXPECT_EQ(z_to_mesh_count, static_cast<int>(builder_config::num_mesh_directions_2d));   // Z_TO_MESH for all mesh directions
    EXPECT_EQ(intra_mesh_count, 0);  // No INTRA_MESH connections
}

// ============================================================================
// Test 5: Multi-VC Connection Scenarios
// ============================================================================

TEST_F(RouterConnectionMappingTest, MultiVC_MeshRouter_VC0_and_VC1) {
    // Mesh routers use VC0 for INTRA_MESH and MESH_TO_Z
    // VC1 receives from Z router (but no outgoing VC1 connections for mesh)

    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::W,
        true);

    // VC0 should have targets (mesh directions + Z)
    bool has_vc0_targets = false;
    for (uint32_t ch = 0; ch < builder_config::num_sender_channels_2d_mesh + 1; ++ch) {
        if (mesh_mapping.has_targets(0, ch)) {
            has_vc0_targets = true;
            break;
        }
    }
    EXPECT_TRUE(has_vc0_targets);

    // VC1 should have no sender targets (mesh doesn't send on VC1)
    bool has_vc1_targets = false;
    for (uint32_t ch = 0; ch < builder_config::num_sender_channels_2d_mesh + 1; ++ch) {
        if (mesh_mapping.has_targets(1, ch)) {
            has_vc1_targets = true;
            break;
        }
    }
    EXPECT_FALSE(has_vc1_targets);
}

TEST_F(RouterConnectionMappingTest, MultiVC_ZRouter_VC0_and_VC1) {
    // Z router uses VC0 for receiving from mesh (no senders)
    // Z router uses VC1 for sending to mesh

    auto z_mapping = RouterConnectionMapping::for_z_router();

    // VC0 should have no sender targets (Z receives on VC0, doesn't send).
    // Z-facing boundary VC0: worker + 4 non-self = 5 channels to check.
    bool has_vc0_targets = false;
    for (uint32_t ch = 0; ch < 5; ++ch) {
        if (z_mapping.has_targets(0, ch)) {
            has_vc0_targets = true;
            break;
        }
    }
    EXPECT_FALSE(has_vc0_targets);

    // VC1 should have sender targets (Z sends on VC1). From-Z fanout = 4 channels.
    bool has_vc1_targets = false;
    for (uint32_t ch = 0; ch < 4; ++ch) {
        if (z_mapping.has_targets(1, ch)) {
            has_vc1_targets = true;
            break;
        }
    }
    EXPECT_TRUE(has_vc1_targets);
}

// ============================================================================
// Test 6: Direction-Based Routing
// ============================================================================

TEST_F(RouterConnectionMappingTest, DirectionBased_ZRouter_ChannelToDirection) {
    // Validate Z router VC1 channel-to-direction mapping
    auto z_mapping = RouterConnectionMapping::for_z_router();

    std::vector<std::pair<uint32_t, RoutingDirection>> expected = {
        {0, RoutingDirection::N},
        {1, RoutingDirection::E},
        {2, RoutingDirection::S},
        {3, RoutingDirection::W}
    };

    // Verify all mesh directions are mapped correctly in receiver channel 0
    TT_FATAL(expected.size() == builder_config::num_mesh_directions_2d, "Test configuration mismatch");

    auto targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), expected.size());

    // Verify each expected direction is present
    for (const auto& [channel_idx, expected_dir] : expected) {
        auto it = std::find_if(targets.begin(), targets.end(), [expected_dir](const ConnectionTarget& t) {
            return t.target_direction.has_value() && t.target_direction.value() == expected_dir;
        });
        ASSERT_NE(it, targets.end());
        EXPECT_EQ(it->target_direction.value(), expected_dir);
    }
}

TEST_F(RouterConnectionMappingTest, DirectionBased_MeshRouter_MeshToZ) {
    // Validate mesh router MESH_TO_Z channel targets Z direction
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::S,
        true);

    auto targets = mesh_mapping.get_downstream_targets(0, 0);
    auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, targets.end());
    EXPECT_EQ(mesh_to_z_it->type, ConnectionType::MESH_TO_Z);
    ASSERT_TRUE(mesh_to_z_it->target_direction.has_value());
    EXPECT_EQ(mesh_to_z_it->target_direction.value(), RoutingDirection::Z);
}

// ============================================================================
// Test 7: Regression Tests (Ensure Old Behavior Preserved)
// ============================================================================

TEST_F(RouterConnectionMappingTest, Regression_MeshToMesh_StillWorks) {
    // Verify that basic mesh-to-mesh connections still work after refactor
    auto mesh_n = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        false);

    auto mesh_s = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::S,
        false);

    // Both should have INTRA_MESH targets on receiver channel 0
    EXPECT_TRUE(mesh_n.has_targets(0, 0));
    EXPECT_TRUE(mesh_s.has_targets(0, 0));

    auto n_targets = mesh_n.get_downstream_targets(0, 0);
    auto s_targets = mesh_s.get_downstream_targets(0, 0);

    EXPECT_GT(n_targets.size(), 0);
    EXPECT_GT(s_targets.size(), 0);

    // All should be INTRA_MESH
    for (const auto& target : n_targets) {
        EXPECT_EQ(target.type, ConnectionType::INTRA_MESH);
    }
    for (const auto& target : s_targets) {
        EXPECT_EQ(target.type, ConnectionType::INTRA_MESH);
    }
}

TEST_F(RouterConnectionMappingTest, Regression_1D_Topology_StillWorks) {
    // Verify 1D topology connections work after refactor
    auto mesh_1d = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::E,
        false);

    // 1D should have fewer targets than 2D
    // Receiver channel 0 should have 1 INTRA_MESH target
    auto targets = mesh_1d.get_downstream_targets(0, 0);
    EXPECT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
}

// ============================================================================
// Test 8: Unified generator -- capability selects the template, direction the slot arithmetic
// ============================================================================

namespace {

// Comparable signature of one mapping: per-VC target lists, order preserved.
using TargetSig = std::tuple<ConnectionType, uint32_t, uint32_t, std::optional<RoutingDirection>>;

std::vector<TargetSig> target_signature(const RouterConnectionMapping& mapping, uint32_t vc) {
    std::vector<TargetSig> out;
    for (const auto& t : mapping.get_downstream_targets(vc, 0)) {
        out.emplace_back(t.type, t.target_vc, t.target_sender_channel, t.target_direction);
    }
    return out;
}

void expect_same_mapping(const RouterConnectionMapping& a, const RouterConnectionMapping& b) {
    for (uint32_t vc : {0u, 1u}) {
        EXPECT_EQ(target_signature(a, vc), target_signature(b, vc)) << "VC" << vc;
    }
}

}  // namespace

TEST_F(RouterConnectionMappingTest, UnifiedGenerator_ZBoundaryMatchesForZRouterAlias) {
    // The unified entry point and the alias must produce the identical boundary template.
    const auto via_unified = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::Z,
        /*has_z=*/false,
        /*enable_vc1=*/true,
        /*enable_mesh_pass_through=*/false,
        /*express_routing_enabled=*/false,
        EdgeCapability::INTERMESH,
        /*has_intramesh_express=*/false);
    const auto via_alias = RouterConnectionMapping::for_z_router();

    expect_same_mapping(via_unified, via_alias);
}

TEST_F(RouterConnectionMappingTest, UnifiedGenerator_ZBoundaryTemplateContents) {
    // The Z-facing intermesh boundary: VC1 from-boundary fanout to every mesh direction, no VC0
    // map (its VC0 senders are fed by the mesh routers' MESH_TO_Z targets, emitted on their maps).
    const auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::Z, false, true, false, false, EdgeCapability::INTERMESH, false);

    EXPECT_FALSE(mapping.has_targets(0, 0));

    const auto targets = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4u);
    std::set<RoutingDirection> directions;
    for (size_t i = 0; i < targets.size(); ++i) {
        EXPECT_EQ(targets[i].type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(targets[i].target_vc, 1u);
        EXPECT_EQ(targets[i].target_sender_channel, i);
        ASSERT_TRUE(targets[i].target_direction.has_value());
        directions.insert(*targets[i].target_direction);
    }
    EXPECT_EQ(
        directions,
        std::set<RoutingDirection>(
            {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}));
}

TEST_F(RouterConnectionMappingTest, UnifiedGenerator_ExpressChordIsOrdinaryMeshLike) {
    // A Z-facing router whose edge is an express chord must not inherit the boundary template:
    // it is wired to all four cardinals as ordinary INTRA_MESH transitions, on VC0 and VC1 alike.
    const auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Torus,
        RoutingDirection::Z,
        /*has_z=*/false,
        /*enable_vc1=*/true,
        /*enable_mesh_pass_through=*/false,
        /*express_routing_enabled=*/true,
        EdgeCapability::INTRAMESH_EXPRESS,
        /*has_intramesh_express=*/true);

    for (uint32_t vc : {0u, 1u}) {
        std::set<RoutingDirection> directions;
        for (const auto& t : mapping.get_downstream_targets(vc, 0)) {
            EXPECT_EQ(t.type, ConnectionType::INTRA_MESH);
            EXPECT_EQ(t.target_vc, vc);
            ASSERT_TRUE(t.target_direction.has_value());
            directions.insert(*t.target_direction);
        }
        EXPECT_EQ(
            directions,
            std::set<RoutingDirection>(
                {RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W}))
            << "VC" << vc;
    }
}

TEST_F(RouterConnectionMappingTest, UnifiedGenerator_CardinalZIsRejected) {
    // An ordinary cardinal-capability Z edge cannot exist: direction letter and capability
    // disagree, which is a configuration error rather than a template to select.
    EXPECT_ANY_THROW(RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::Z, false, true, false, false, EdgeCapability::INTRAMESH_CARDINAL, false));
}
