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
    void verify_target(
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

    // Z router has 4 sender channels on VC1
    EXPECT_EQ(keys.size(), 4);

    // Verify all keys are VC1, channels 0-3
    for (const auto& key : keys) {
        EXPECT_EQ(key.vc, 1);
        EXPECT_GE(key.receiver_channel, 0);
        EXPECT_LE(key.receiver_channel, 3);
    }
}

// ============================================================================
// Mesh Router Tests - 1D Topology
// ============================================================================

TEST_F(RouterConnectionMappingTest, MeshRouter_1D_NoZ_SingleConnection) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::N,
        false);  // No Z router

    // Channel 0: Reserved (no targets)
    EXPECT_FALSE(mapping.has_targets(0, 0));

    // Channel 1: Connects to opposite direction (SOUTH)
    ASSERT_TRUE(mapping.has_targets(0, 1));
    auto targets = mapping.get_downstream_targets(0, 1);
    ASSERT_EQ(targets.size(), 1);

    verify_target(targets[0], ConnectionType::INTRA_MESH, 0, RoutingDirection::S);
}

TEST_F(RouterConnectionMappingTest, MeshRouter_1D_WithZ_TwoConnections) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::E,
        true);  // Has Z router

    // Channel 1: INTRA_MESH to opposite direction (WEST)
    auto targets_ch1 = mapping.get_downstream_targets(0, 1);
    ASSERT_EQ(targets_ch1.size(), 1);
    verify_target(targets_ch1[0], ConnectionType::INTRA_MESH, 0, RoutingDirection::W);

    // Channel 2: MESH_TO_Z connection
    auto targets_ch2 = mapping.get_downstream_targets(0, 2);
    ASSERT_EQ(targets_ch2.size(), 1);
    verify_target(targets_ch2[0], ConnectionType::MESH_TO_Z, 0, RoutingDirection::Z);
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

        auto targets = mapping.get_downstream_targets(0, 1);
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

    // Channel 0: Reserved (no targets)
    EXPECT_FALSE(mapping.has_targets(0, 0));

    // Channels 1-3: Connect to 3 directions (opposite + 2 cross)
    // NORTH router connects to: SOUTH (ch1), EAST (ch2), WEST (ch3)

    auto targets_ch1 = mapping.get_downstream_targets(0, 1);
    ASSERT_EQ(targets_ch1.size(), 1);
    verify_target(targets_ch1[0], ConnectionType::INTRA_MESH, 0, RoutingDirection::S);

    auto targets_ch2 = mapping.get_downstream_targets(0, 2);
    ASSERT_EQ(targets_ch2.size(), 1);
    verify_target(targets_ch2[0], ConnectionType::INTRA_MESH, 0);
    // Cross direction (EAST or WEST)
    EXPECT_TRUE(
        targets_ch2[0].target_direction == RoutingDirection::E ||
        targets_ch2[0].target_direction == RoutingDirection::W);

    auto targets_ch3 = mapping.get_downstream_targets(0, 3);
    ASSERT_EQ(targets_ch3.size(), 1);
    verify_target(targets_ch3[0], ConnectionType::INTRA_MESH, 0);
    // Other cross direction
    EXPECT_TRUE(
        targets_ch3[0].target_direction == RoutingDirection::E ||
        targets_ch3[0].target_direction == RoutingDirection::W);

    // Verify channels 2 and 3 have different directions
    EXPECT_NE(targets_ch2[0].target_direction, targets_ch3[0].target_direction);
}

TEST_F(RouterConnectionMappingTest, MeshRouter_2D_WithZ_FourConnections) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::E,
        true);  // Has Z router

    // Channels 1-3: INTRA_MESH
    EXPECT_TRUE(mapping.has_targets(0, 1));
    EXPECT_TRUE(mapping.has_targets(0, 2));
    EXPECT_TRUE(mapping.has_targets(0, 3));

    // Channel 4: MESH_TO_Z connection
    auto targets_ch4 = mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(targets_ch4.size(), 1);
    verify_target(targets_ch4[0], ConnectionType::MESH_TO_Z, 0, RoutingDirection::Z);
}

TEST_F(RouterConnectionMappingTest, MeshRouter_2D_EastRouter_CorrectDirections) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::E,
        false);

    // EAST router should connect to: WEST (opposite), NORTH, SOUTH
    auto targets_ch1 = mapping.get_downstream_targets(0, 1);
    ASSERT_EQ(targets_ch1.size(), 1);
    verify_target(targets_ch1[0], ConnectionType::INTRA_MESH, 0, RoutingDirection::W);

    // Channels 2-3 should be NORTH and SOUTH (in some order)
    auto targets_ch2 = mapping.get_downstream_targets(0, 2);
    auto targets_ch3 = mapping.get_downstream_targets(0, 3);

    std::set<RoutingDirection> cross_dirs = {
        targets_ch2[0].target_direction.value(),
        targets_ch3[0].target_direction.value()
    };

    EXPECT_TRUE(cross_dirs.count(RoutingDirection::N));
    EXPECT_TRUE(cross_dirs.count(RoutingDirection::S));
}

// ============================================================================
// Z Router Tests
// ============================================================================

TEST_F(RouterConnectionMappingTest, ZRouter_VC1_FourSenderChannels) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // VC0: No connections (reserved for future use)
    EXPECT_FALSE(mapping.has_targets(0, 0));
    EXPECT_FALSE(mapping.has_targets(0, 1));

    // VC1: 4 sender channels (0-3)
    EXPECT_TRUE(mapping.has_targets(1, 0));
    EXPECT_TRUE(mapping.has_targets(1, 1));
    EXPECT_TRUE(mapping.has_targets(1, 2));
    EXPECT_TRUE(mapping.has_targets(1, 3));

    // Total sender count = 4 (VC1 channels 0-3)
    EXPECT_EQ(mapping.get_total_sender_count(), 4);
}

TEST_F(RouterConnectionMappingTest, ZRouter_VC1_CorrectDirectionMapping) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // Sender channel 0 → NORTH
    auto targets_ch0 = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets_ch0.size(), 1);
    verify_target(targets_ch0[0], ConnectionType::Z_TO_MESH, 1, RoutingDirection::N);

    // Sender channel 1 → EAST
    auto targets_ch1 = mapping.get_downstream_targets(1, 1);
    ASSERT_EQ(targets_ch1.size(), 1);
    verify_target(targets_ch1[0], ConnectionType::Z_TO_MESH, 1, RoutingDirection::E);

    // Sender channel 2 → SOUTH
    auto targets_ch2 = mapping.get_downstream_targets(1, 2);
    ASSERT_EQ(targets_ch2.size(), 1);
    verify_target(targets_ch2[0], ConnectionType::Z_TO_MESH, 1, RoutingDirection::S);

    // Sender channel 3 → WEST
    auto targets_ch3 = mapping.get_downstream_targets(1, 3);
    ASSERT_EQ(targets_ch3.size(), 1);
    verify_target(targets_ch3[0], ConnectionType::Z_TO_MESH, 1, RoutingDirection::W);
}

TEST_F(RouterConnectionMappingTest, ZRouter_AllTargets_SameVC) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // All Z router VC1 targets should go to mesh router VC1
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(1, ch);
        ASSERT_EQ(targets.size(), 1);
        EXPECT_EQ(targets[0].target_vc, 1);  // Z VC1 → mesh VC1
        EXPECT_EQ(targets[0].type, ConnectionType::Z_TO_MESH);
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

    // Verify each mesh router has MESH_TO_Z on channel 4
    EXPECT_TRUE(north_mapping.has_targets(0, 4));
    EXPECT_TRUE(east_mapping.has_targets(0, 4));
    EXPECT_TRUE(south_mapping.has_targets(0, 4));
    EXPECT_TRUE(west_mapping.has_targets(0, 4));
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

    // Mesh routers still have MESH_TO_Z connections
    auto north_targets = north_mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(north_targets.size(), 1);
    EXPECT_EQ(north_targets[0].type, ConnectionType::MESH_TO_Z);

    auto east_targets = east_mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(east_targets.size(), 1);
    EXPECT_EQ(east_targets[0].type, ConnectionType::MESH_TO_Z);
}

TEST_F(RouterConnectionMappingTest, Scenario_1DMesh_NoZ_Bidirectional) {
    // Simple 1D mesh with 2 routers (NORTH and SOUTH)
    auto north_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::N, false);
    auto south_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::S, false);

    // NORTH router channel 1 → SOUTH
    auto north_targets = north_mapping.get_downstream_targets(0, 1);
    ASSERT_EQ(north_targets.size(), 1);
    EXPECT_EQ(north_targets[0].target_direction.value(), RoutingDirection::S);

    // SOUTH router channel 1 → NORTH
    auto south_targets = south_mapping.get_downstream_targets(0, 1);
    ASSERT_EQ(south_targets.size(), 1);
    EXPECT_EQ(south_targets[0].target_direction.value(), RoutingDirection::N);
}

// ============================================================================
// Negative Tests: Invalid Channel Access and Overflow
// ============================================================================

TEST_F(RouterConnectionMappingTest, ZRouter_VC1_OnlyFourDirections) {
    // Validate that Z router VC1 only defines connections for 4 directions (N/E/S/W)
    auto mapping = RouterConnectionMapping::for_z_router();

    // VC1 should have exactly 4 sender channels (0-3)
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(1, ch);
        EXPECT_EQ(targets.size(), 1) << "Z VC1 sender " << ch << " should have 1 target";
    }

    // Channel 4 and beyond should have no targets
    for (uint32_t ch = 4; ch < 10; ++ch) {
        auto targets = mapping.get_downstream_targets(1, ch);
        EXPECT_TRUE(targets.empty()) << "Z VC1 sender " << ch << " should have no targets (doesn't exist)";
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
    // Mesh router VC0 has 4 sender channels (0-3), querying beyond should return empty
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::N, false);

    // Channel 0 is reserved for local/internal use (no INTRA_MESH targets)
    EXPECT_FALSE(mapping.has_targets(0, 0));

    // Channels 1-3 are forwarding channels (should have targets)
    EXPECT_TRUE(mapping.has_targets(0, 1));
    EXPECT_TRUE(mapping.has_targets(0, 2));
    EXPECT_TRUE(mapping.has_targets(0, 3));

    // Invalid channels (beyond 3) should have no targets
    for (uint32_t ch = 10; ch < 20; ++ch) {
        auto targets = mapping.get_downstream_targets(0, ch);
        EXPECT_TRUE(targets.empty()) << "Mesh router VC0 channel " << ch << " should have no targets (doesn't exist)";
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
