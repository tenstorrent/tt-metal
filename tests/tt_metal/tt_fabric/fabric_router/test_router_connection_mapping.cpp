// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"

using namespace tt::tt_fabric;

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
    
    auto keys = mapping.get_all_sender_keys();
    
    // Z router has 4 sender channels on VC1
    EXPECT_EQ(keys.size(), 4);
    
    // Verify all keys are VC1, channels 0-3
    for (const auto& key : keys) {
        EXPECT_EQ(key.vc, 1);
        EXPECT_GE(key.sender_channel, 0);
        EXPECT_LE(key.sender_channel, 3);
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
    
    // Channel 2: MESH_TO_Z aggregation
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
            << "Direction " << static_cast<int>(my_dir) 
            << " should connect to " << static_cast<int>(expected_opposite);
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
    
    // Channel 4: MESH_TO_Z aggregation
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

TEST_F(RouterConnectionMappingTest, SenderChannelKey_Comparison) {
    SenderChannelKey key1{0, 1};
    SenderChannelKey key2{0, 2};
    SenderChannelKey key3{1, 0};
    SenderChannelKey key4{0, 1};
    
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
    // Phase 5 orchestration will skip SOUTH and WEST
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

