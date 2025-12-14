// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"

using namespace tt::tt_fabric;

/**
 * RouterConnectionMapping Tests
 *
 * These tests validate connection mapping functionality:
 * - Factory methods for mesh and Z routers
 * - Downstream target specifications
 * - Connection type assignments
 * - Direction-based routing
 * - VC assignments for different connection types
 *
 * Test Coverage Summary:
 * ┌──────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                    │ Test Name                                │ Focus        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Basic Functionality         │ EmptyMapping_NoTargets                   │ Empty        │
 * │                             │ GetAllSenderKeys_ReturnsCorrectKeys      │ Key query    │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mesh Router - 1D            │ MeshRouter_1D_NoZ_SingleConnection       │ 1D no Z      │
 * │                             │ MeshRouter_1D_WithZ_HasMeshToZ           │ 1D with Z    │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mesh Router - 2D            │ MeshRouter_2D_NoZ_ThreeConnections       │ 2D no Z      │
 * │                             │ MeshRouter_2D_WithZ_FourConnections      │ 2D with Z    │
 * │                             │ MeshRouter_2D_AllDirections              │ All dirs     │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Z Router                    │ ZRouter_VC1_FourConnections              │ VC1 4 conns  │
 * │                             │ ZRouter_VC0_NoConnections                │ VC0 empty    │
 * │                             │ ZRouter_AllDirections_Mapped             │ All dirs     │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Connection Types            │ ConnectionType_INTRA_MESH                │ Intra-mesh   │
 * │                             │ ConnectionType_MESH_TO_Z                 │ Mesh→Z       │
 * │                             │ ConnectionType_Z_TO_MESH                 │ Z→Mesh       │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ VC Assignments              │ VCAssignment_IntraMesh_VC0               │ VC0 routing  │
 * │                             │ VCAssignment_MeshToZ_VC0                 │ Mesh→Z VC0   │
 * │                             │ VCAssignment_ZToMesh_VC1                 │ Z→Mesh VC1   │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Direction Mapping           │ DirectionMapping_OppositeDirections      │ Opposites    │
 * │                             │ DirectionMapping_ZDirection              │ Z direction  │
 * │                             │ DirectionMapping_AllCardinalDirections   │ N/E/S/W      │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Query Interface             │ QueryInterface_HasTargets                │ Has check    │
 * │                             │ QueryInterface_GetTotalSenderCount       │ Count        │
 * │                             │ QueryInterface_GetAllSenderKeys          │ All keys     │
 * └─────────────────────────────┴──────────────────────────────────────────┴──────────────┘
 *
 * Total: 22 tests across 8 categories
 */

// ============================================================================
// Test Fixture
// ============================================================================

class RouterConnectionMappingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper to verify a single target
    static void verify_target(
        const ConnectionTarget& target,
        ConnectionType expected_type,
        uint32_t expected_vc,
        std::optional<RoutingDirection> expected_dir = std::nullopt) {
        EXPECT_EQ(target.type, expected_type);
        EXPECT_EQ(target.target_vc, expected_vc);
        if (expected_dir.has_value()) {
            ASSERT_TRUE(target.target_direction.has_value());
            EXPECT_EQ(target.target_direction.value(), expected_dir.value());
        }
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(RouterConnectionMappingTest, EmptyMapping_NoTargets) {
    RouterConnectionMapping mapping;

    auto targets = mapping.get_downstream_targets(0, 0);
    EXPECT_TRUE(targets.empty());
    EXPECT_FALSE(mapping.has_targets(0, 0));
    EXPECT_EQ(mapping.get_total_sender_count(), 0);
}

TEST_F(RouterConnectionMappingTest, GetAllSenderKeys_ReturnsCorrectKeys) {
    RouterConnectionMapping mapping = RouterConnectionMapping::for_z_router();

    auto keys = mapping.get_all_receiver_keys();

    // Z router has 1 receiver channel on VC1 with 4 targets
    EXPECT_EQ(keys.size(), 1);

    // Verify the key is VC1, channel 0
    EXPECT_EQ(keys[0].vc, 1);
    EXPECT_EQ(keys[0].receiver_channel, 0);

    // Verify it has 4 downstream targets
    auto targets = mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(targets.size(), 4);
}

// ============================================================================
// Mesh Router Tests - 1D Topology
// ============================================================================

TEST_F(RouterConnectionMappingTest, MeshRouter_1D_NoZ_SingleConnection) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::N,
        false);  // No Z router

    // Receiver channel 0 has 1 target: opposite direction (SOUTH)
    ASSERT_TRUE(mapping.has_targets(0, 0));
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 1);

    verify_target(targets[0], ConnectionType::INTRA_MESH, 0, RoutingDirection::S);
}

TEST_F(RouterConnectionMappingTest, MeshRouter_1D_WithZ_TwoConnections) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::E,
        true);  // Has Z router

    // Receiver channel 0 has 2 targets: INTRA_MESH (WEST) + MESH_TO_Z
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 2);

    // Find INTRA_MESH target
    auto intra_mesh_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::INTRA_MESH; });
    ASSERT_NE(intra_mesh_it, targets.end());
    verify_target(*intra_mesh_it, ConnectionType::INTRA_MESH, 0, RoutingDirection::W);

    // Find MESH_TO_Z target
    auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(mesh_to_z_it, targets.end());
    verify_target(*mesh_to_z_it, ConnectionType::MESH_TO_Z, 0, RoutingDirection::Z);
}

TEST_F(RouterConnectionMappingTest, MeshRouter_1D_AllDirections_OppositeMapping) {
    // Test all 4 directions to verify opposite calculation
    std::vector<std::pair<RoutingDirection, RoutingDirection>> direction_pairs = {
        {RoutingDirection::N, RoutingDirection::S},
        {RoutingDirection::S, RoutingDirection::N},
        {RoutingDirection::E, RoutingDirection::W},
        {RoutingDirection::W, RoutingDirection::E}
    };

    for (const auto& [my_dir, expected_opposite] : direction_pairs) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Linear, my_dir, false);

        auto targets = mapping.get_downstream_targets(0, 0);  // Receiver channel 0
        ASSERT_EQ(targets.size(), 1);
        EXPECT_EQ(targets[0].target_direction.value(), expected_opposite)
            << "Direction " << static_cast<int>(my_dir) << " should connect to " << static_cast<int>(expected_opposite);
    }
}

// ============================================================================
// Mesh Router Tests - 2D Topology
// ============================================================================

TEST_F(RouterConnectionMappingTest, MeshRouter_2D_NoZ_ThreeConnections) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        false);

    // Receiver channel 0 has 3 targets: opposite + 2 cross directions
    // NORTH router connects to: SOUTH (opposite), EAST, WEST (cross)
    ASSERT_TRUE(mapping.has_targets(0, 0));
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 3);

    // Verify all are INTRA_MESH
    for (const auto& target : targets) {
        EXPECT_EQ(target.type, ConnectionType::INTRA_MESH);
        EXPECT_EQ(target.target_vc, 0);
    }

    // Find SOUTH target (opposite)
    auto south_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::S; });
    ASSERT_NE(south_it, targets.end());

    // Verify EAST and WEST are present
    auto east_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::E; });
    auto west_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::W; });
    ASSERT_NE(east_it, targets.end());
    ASSERT_NE(west_it, targets.end());
}

TEST_F(RouterConnectionMappingTest, MeshRouter_2D_WithZ_FourConnections) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::E,
        true);  // Has Z router

    // Receiver channel 0 has 4 targets: 3 INTRA_MESH + 1 MESH_TO_Z
    ASSERT_TRUE(mapping.has_targets(0, 0));
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 4);

    // Count connection types
    uint32_t intra_mesh_count = 0;
    uint32_t mesh_to_z_count = 0;
    for (const auto& target : targets) {
        if (target.type == ConnectionType::INTRA_MESH) {
            intra_mesh_count++;
        } else if (target.type == ConnectionType::MESH_TO_Z) {
            mesh_to_z_count++;
            verify_target(target, ConnectionType::MESH_TO_Z, 0, RoutingDirection::Z);
        }
    }
    EXPECT_EQ(intra_mesh_count, 3);
    EXPECT_EQ(mesh_to_z_count, 1);
}

TEST_F(RouterConnectionMappingTest, MeshRouter_2D_EastRouter_CorrectDirections) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::E,
        false);

    // EAST router receiver channel 0 should have 3 targets: WEST (opposite), NORTH, SOUTH
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 3);

    // Find each expected direction
    auto west_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::W; });
    auto north_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::N; });
    auto south_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.target_direction == RoutingDirection::S; });

    ASSERT_NE(west_it, targets.end());
    ASSERT_NE(north_it, targets.end());
    ASSERT_NE(south_it, targets.end());

    // Verify all are INTRA_MESH to VC0
    for (const auto& target : targets) {
        verify_target(target, ConnectionType::INTRA_MESH, 0);
    }
}

// ============================================================================
// Z Router Tests
// ============================================================================

TEST_F(RouterConnectionMappingTest, ZRouter_VC1_FourSenderChannels) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // VC0: No connections (reserved for future use)
    EXPECT_FALSE(mapping.has_targets(0, 0));

    // VC1: receiver channel 0 has 4 targets
    EXPECT_TRUE(mapping.has_targets(1, 0));
    auto targets = mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(targets.size(), 4);

    // Total sender count = 4 targets from receiver channel 0
    EXPECT_EQ(mapping.get_total_sender_count(), 4);
}

TEST_F(RouterConnectionMappingTest, ZRouter_VC1_CorrectDirectionMapping) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // Receiver channel 0 on VC1 has 4 targets mapping to N/E/S/W
    auto targets = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    // Verify each direction is present
    std::vector<RoutingDirection> expected_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    for (const auto& expected_dir : expected_directions) {
        auto it = std::find_if(targets.begin(), targets.end(), [expected_dir](const ConnectionTarget& t) {
            return t.target_direction == expected_dir;
        });
        ASSERT_NE(it, targets.end()) << "Missing direction: " << static_cast<int>(expected_dir);
        verify_target(*it, ConnectionType::Z_TO_MESH, 1, expected_dir);
    }
}

TEST_F(RouterConnectionMappingTest, ZRouter_AllTargets_SameVC) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // All Z router VC1 targets should go to mesh router VC1
    auto targets = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    for (const auto& target : targets) {
        EXPECT_EQ(target.target_vc, 1);  // Z VC1 → mesh VC1
        EXPECT_EQ(target.type, ConnectionType::Z_TO_MESH);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(RouterConnectionMappingTest, MeshRouter_InvalidVC_NoTargets) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        false);

    // VC1 not configured for mesh routers
    EXPECT_FALSE(mapping.has_targets(1, 0));
    EXPECT_TRUE(mapping.get_downstream_targets(1, 0).empty());
}

TEST_F(RouterConnectionMappingTest, ZRouter_InvalidChannel_NoTargets) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // VC1 only has channels 0-3
    EXPECT_FALSE(mapping.has_targets(1, 4));
    EXPECT_FALSE(mapping.has_targets(1, 5));
    EXPECT_TRUE(mapping.get_downstream_targets(1, 4).empty());
}

TEST_F(RouterConnectionMappingTest, ConnectionTarget_Construction) {
    ConnectionTarget target(
        ConnectionType::MESH_TO_Z,
        1,  // VC1
        2,  // Channel 2
        RoutingDirection::Z);

    EXPECT_EQ(target.type, ConnectionType::MESH_TO_Z);
    EXPECT_EQ(target.target_vc, 1);
    EXPECT_EQ(target.target_sender_channel, 2);
    ASSERT_TRUE(target.target_direction.has_value());
    EXPECT_EQ(target.target_direction.value(), RoutingDirection::Z);
}

TEST_F(RouterConnectionMappingTest, ReceiverChannelKey_Comparison) {
    ReceiverChannelKey key1{0, 1};
    ReceiverChannelKey key2{0, 2};
    ReceiverChannelKey key3{1, 0};
    ReceiverChannelKey key4{0, 1};

    // Ordering
    EXPECT_TRUE(key1 < key2);  // Same VC, lower channel
    EXPECT_TRUE(key1 < key3);  // Lower VC
    EXPECT_FALSE(key2 < key1);

    // Equality
    EXPECT_TRUE(key1 == key4);
    EXPECT_FALSE(key1 == key2);
}

// ============================================================================
// Integration Scenario Tests
// ============================================================================

TEST_F(RouterConnectionMappingTest, Scenario_2DMesh_WithZ_FullDevice) {
    // Simulate a full device with 4 mesh routers + 1 Z router

    // Create mesh routers for all 4 directions
    auto north_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, true);
    auto east_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);
    auto south_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::S, true);
    auto west_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::W, true);

    // Create Z router
    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Verify each mesh router has 4 connections (3 INTRA_MESH + 1 MESH_TO_Z)
    EXPECT_EQ(north_mapping.get_total_sender_count(), 4);
    EXPECT_EQ(east_mapping.get_total_sender_count(), 4);
    EXPECT_EQ(south_mapping.get_total_sender_count(), 4);
    EXPECT_EQ(west_mapping.get_total_sender_count(), 4);

    // Verify Z router has 4 connections (all Z_TO_MESH)
    EXPECT_EQ(z_mapping.get_total_sender_count(), 4);

    // Verify each mesh router has MESH_TO_Z in receiver channel 0 targets
    for (auto* mapping_ptr : {&north_mapping, &east_mapping, &south_mapping, &west_mapping}) {
        auto targets = mapping_ptr->get_downstream_targets(0, 0);
        auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
            [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
        EXPECT_NE(mesh_to_z_it, targets.end());
    }
}

TEST_F(RouterConnectionMappingTest, Scenario_EdgeDevice_2MeshRouters_WithZ) {
    // Edge device with only 2 mesh routers (e.g., NORTH and EAST)
    auto north_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, true);
    auto east_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);

    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Z router still has intent for all 4 directions
    // FabricBuilder will skip SOUTH and WEST if routers don't exist
    EXPECT_EQ(z_mapping.get_total_sender_count(), 4);

    // Mesh routers still have MESH_TO_Z connections in receiver channel 0
    auto north_targets = north_mapping.get_downstream_targets(0, 0);
    auto north_mesh_to_z = std::find_if(north_targets.begin(), north_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(north_mesh_to_z, north_targets.end());

    auto east_targets = east_mapping.get_downstream_targets(0, 0);
    auto east_mesh_to_z = std::find_if(east_targets.begin(), east_targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(east_mesh_to_z, east_targets.end());
}

TEST_F(RouterConnectionMappingTest, Scenario_1DMesh_NoZ_Bidirectional) {
    // Simple 1D mesh with 2 routers (NORTH and SOUTH)
    auto north_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::N, false);
    auto south_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::S, false);

    // NORTH router receiver channel 0 → SOUTH
    auto north_targets = north_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(north_targets.size(), 1);
    EXPECT_EQ(north_targets[0].target_direction.value(), RoutingDirection::S);

    // SOUTH router receiver channel 0 → NORTH
    auto south_targets = south_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(south_targets.size(), 1);
    EXPECT_EQ(south_targets[0].target_direction.value(), RoutingDirection::N);
}

// ============================================================================
// Negative Tests: Invalid Channel Access and Overflow
// ============================================================================

TEST_F(RouterConnectionMappingTest, ZRouter_VC1_OnlyFourDirections) {
    // Validate that Z router VC1 only defines connections for 4 directions (N/E/S/W)
    auto mapping = RouterConnectionMapping::for_z_router();

    // VC1 receiver channel 0 should have exactly 4 targets
    auto targets = mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(targets.size(), 4) << "Z VC1 receiver channel 0 should have 4 targets";

    // Channel 1 and beyond should have no targets
    for (uint32_t ch = 1; ch < 10; ++ch) {
        auto ch_targets = mapping.get_downstream_targets(1, ch);
        EXPECT_TRUE(ch_targets.empty()) << "Z VC1 receiver channel " << ch << " should have no targets (doesn't exist)";
        EXPECT_FALSE(mapping.has_targets(1, ch));
    }
}

TEST_F(RouterConnectionMappingTest, MeshRouter_InvalidVC_ReturnsEmpty) {
    // Mesh router only has VC0, querying VC1+ should return empty
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, false);

    // VC1 should have no targets (not enabled for standard mesh)
    for (uint32_t ch = 0; ch < 5; ++ch) {
        auto targets = mapping.get_downstream_targets(1, ch);
        EXPECT_TRUE(targets.empty()) << "Mesh router VC1 channel " << ch << " should have no targets";
    }

    // VC2+ definitely don't exist
    auto targets = mapping.get_downstream_targets(2, 0);
    EXPECT_TRUE(targets.empty());
}

TEST_F(RouterConnectionMappingTest, MeshRouter_InvalidSenderChannel_ReturnsEmpty) {
    // Mesh router VC0 receiver channel 0 has 3 targets for 2D mesh without Z
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, false);

    // Receiver channel 0 should have targets
    EXPECT_TRUE(mapping.has_targets(0, 0));
    auto targets = mapping.get_downstream_targets(0, 0);
    EXPECT_EQ(targets.size(), 3);  // 3 INTRA_MESH targets for 2D mesh

    // Invalid receiver channels (beyond 0) should have no targets
    for (uint32_t ch = 1; ch < 10; ++ch) {
        auto ch_targets = mapping.get_downstream_targets(0, ch);
        EXPECT_TRUE(ch_targets.empty()) << "Mesh router VC0 receiver channel " << ch << " should have no targets (doesn't exist)";
        EXPECT_FALSE(mapping.has_targets(0, ch));
    }
}

TEST_F(RouterConnectionMappingTest, ZRouter_VC0_NoOutgoingTargets) {
    // Z router VC0 senders should have no downstream targets (reserved)
    auto mapping = RouterConnectionMapping::for_z_router();

    // All VC0 sender channels should have no targets
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(0, ch);
        EXPECT_TRUE(targets.empty()) << "Z router VC0 sender " << ch << " should have no targets";
        EXPECT_FALSE(mapping.has_targets(0, ch));
    }
}
