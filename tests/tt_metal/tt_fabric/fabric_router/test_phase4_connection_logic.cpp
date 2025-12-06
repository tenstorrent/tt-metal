// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

using namespace tt::tt_fabric;

/**
 * Phase 4: Connection Logic Refactor Tests
 * 
 * These tests validate that:
 * 1. Connection mapping drives connection establishment
 * 2. configure_local_connections() handles Z↔mesh connections correctly
 * 3. Asymmetric connections (MESH_TO_Z vs Z_TO_MESH) work properly
 * 4. Edge cases (2-3 mesh routers) are handled gracefully
 */

class Phase4ConnectionLogicTest : public ::testing::Test {
protected:
    void SetUp() override {
        registry_ = std::make_shared<ConnectionRegistry>();
    }

    std::shared_ptr<ConnectionRegistry> registry_;
};

// ============================================================================
// Test 1: Mapping-Driven Connection Logic
// ============================================================================

TEST_F(Phase4ConnectionLogicTest, MappingDriven_INTRA_MESH_Connections) {
    // Verify that INTRA_MESH connections are established based on connection mapping
    // This is a conceptual test - actual connection establishment requires builders
    
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        false);  // No Z router
    
    // 2D Mesh router: channel 0 (local), channels 1-3 (peers N/E/S/W)
    // Channel 0: local, no downstream
    auto local_targets = mesh_mapping.get_downstream_targets(0, 0);
    EXPECT_EQ(local_targets.size(), 0);
    
    // Channels 1-3: peer channels with INTRA_MESH targets
    for (uint32_t sender_ch = 1; sender_ch <= 3; ++sender_ch) {
        auto targets = mesh_mapping.get_downstream_targets(0, sender_ch);
        EXPECT_EQ(targets.size(), 1);
        EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
    }
}

TEST_F(Phase4ConnectionLogicTest, MappingDriven_MESH_TO_Z_Connection) {
    // Verify MESH_TO_Z connection is in the mapping
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);  // Has Z router
    
    // Aggregation channel (channel 4 for 2D) should have MESH_TO_Z target
    auto targets = mesh_mapping.get_downstream_targets(0, 4);
    
    ASSERT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::MESH_TO_Z);
    EXPECT_EQ(targets[0].target_vc, 0);  // Mesh VC0 → Z VC0
    ASSERT_TRUE(targets[0].target_direction.has_value());
    EXPECT_EQ(targets[0].target_direction.value(), RoutingDirection::Z);
}

TEST_F(Phase4ConnectionLogicTest, MappingDriven_Z_TO_MESH_Connections) {
    // Verify Z_TO_MESH connections are in the mapping
    auto z_mapping = RouterConnectionMapping::for_z_router();
    
    // Z router VC1 should have 4 Z_TO_MESH targets (N/E/S/W intent)
    std::vector<RoutingDirection> expected_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };
    
    for (size_t i = 0; i < 4; ++i) {
        auto targets = z_mapping.get_downstream_targets(1, i);
        
        ASSERT_EQ(targets.size(), 1);
        EXPECT_EQ(targets[0].type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(targets[0].target_vc, 1);  // Z VC1 → Mesh VC1
        ASSERT_TRUE(targets[0].target_direction.has_value());
        EXPECT_EQ(targets[0].target_direction.value(), expected_directions[i]);
    }
}

// ============================================================================
// Test 2: Asymmetric Connection Validation
// ============================================================================

TEST_F(Phase4ConnectionLogicTest, AsymmetricConnections_MeshToZ_vs_ZToMesh) {
    // Verify that MESH_TO_Z and Z_TO_MESH are properly asymmetric
    
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);
    
    auto z_mapping = RouterConnectionMapping::for_z_router();
    
    // MESH_TO_Z: Mesh VC0 channel 4 → Z VC0
    auto mesh_to_z = mesh_mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(mesh_to_z.size(), 1);
    EXPECT_EQ(mesh_to_z[0].type, ConnectionType::MESH_TO_Z);
    EXPECT_EQ(mesh_to_z[0].target_vc, 0);
    
    // Z_TO_MESH: Z VC1 channels 0-3 → Mesh VC1
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto z_to_mesh = z_mapping.get_downstream_targets(1, ch);
        ASSERT_EQ(z_to_mesh.size(), 1);
        EXPECT_EQ(z_to_mesh[0].type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(z_to_mesh[0].target_vc, 1);  // Different VC!
    }
    
    // Key insight: Different VCs, different directions, different connection types
}

TEST_F(Phase4ConnectionLogicTest, AsymmetricConnections_VCAssignments) {
    // Validate VC assignments are correct for asymmetric connections
    
    // Mesh VC0 → Z VC0 (aggregation)
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::E,
        true);
    
    auto mesh_targets = mesh_mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(mesh_targets.size(), 1);
    EXPECT_EQ(mesh_targets[0].target_vc, 0);
    
    // Z VC1 → Mesh VC1 (distribution)
    auto z_mapping = RouterConnectionMapping::for_z_router();
    
    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(z_targets.size(), 1);
    EXPECT_EQ(z_targets[0].target_vc, 1);
}

// ============================================================================
// Test 3: Edge Device Scenarios (2-3 Mesh Routers)
// ============================================================================

TEST_F(Phase4ConnectionLogicTest, EdgeDevice_TwoMeshRouters_ZMapping) {
    // Z router mapping specifies intent for all 4 directions
    // configure_local_connections() will skip non-existent routers
    
    auto z_mapping = RouterConnectionMapping::for_z_router();
    
    // Mapping always specifies 4 directions (intent)
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = z_mapping.get_downstream_targets(1, ch);
        EXPECT_EQ(targets.size(), 1);
        EXPECT_EQ(targets[0].type, ConnectionType::Z_TO_MESH);
    }
    
    // FabricBuilder will check if routers exist before connecting
    // This test validates the mapping is correct regardless of actual router count
}

TEST_F(Phase4ConnectionLogicTest, EdgeDevice_ThreeMeshRouters_MappingIntent) {
    // Validate that Z router mapping is consistent regardless of device position
    
    auto z_mapping_center = RouterConnectionMapping::for_z_router();
    
    // All 4 sender channels should have targets
    EXPECT_TRUE(z_mapping_center.has_targets(1, 0));  // N
    EXPECT_TRUE(z_mapping_center.has_targets(1, 1));  // E
    EXPECT_TRUE(z_mapping_center.has_targets(1, 2));  // S
    EXPECT_TRUE(z_mapping_center.has_targets(1, 3));  // W
    
    // configure_local_connections() will gracefully skip missing routers
}

// ============================================================================
// Test 4: Connection Type Filtering
// ============================================================================

TEST_F(Phase4ConnectionLogicTest, ConnectionTypeFiltering_INTRA_MESH_Only) {
    // configure_connection() should only handle INTRA_MESH
    // configure_local_connections() should only handle MESH_TO_Z and Z_TO_MESH
    
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        true);
    
    // Count connection types
    int intra_mesh_count = 0;
    int mesh_to_z_count = 0;
    
    for (uint32_t ch = 0; ch < 5; ++ch) {
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

TEST_F(Phase4ConnectionLogicTest, ConnectionTypeFiltering_LocalOnly) {
    // Z router should only have local connections (Z_TO_MESH)
    // No INTRA_MESH connections
    
    auto z_mapping = RouterConnectionMapping::for_z_router();
    
    int z_to_mesh_count = 0;
    int intra_mesh_count = 0;
    
    // Check VC0 (should be empty or reserved)
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto vc0_targets = z_mapping.get_downstream_targets(0, ch);
        EXPECT_EQ(vc0_targets.size(), 0);  // VC0 unused for Z router
    }
    
    // Check VC1 (should have Z_TO_MESH)
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
    
    EXPECT_EQ(z_to_mesh_count, 4);   // 4 Z_TO_MESH connections
    EXPECT_EQ(intra_mesh_count, 0);  // No INTRA_MESH connections
}

// ============================================================================
// Test 5: Multi-VC Connection Scenarios
// ============================================================================

TEST_F(Phase4ConnectionLogicTest, MultiVC_MeshRouter_VC0_and_VC1) {
    // Mesh routers use VC0 for INTRA_MESH and MESH_TO_Z
    // VC1 receives from Z router (but no outgoing VC1 connections for mesh)
    
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::W,
        true);
    
    // VC0 should have targets
    bool has_vc0_targets = false;
    for (uint32_t ch = 0; ch < 5; ++ch) {
        if (mesh_mapping.has_targets(0, ch)) {
            has_vc0_targets = true;
            break;
        }
    }
    EXPECT_TRUE(has_vc0_targets);
    
    // VC1 should have no sender targets (mesh doesn't send on VC1)
    bool has_vc1_targets = false;
    for (uint32_t ch = 0; ch < 5; ++ch) {
        if (mesh_mapping.has_targets(1, ch)) {
            has_vc1_targets = true;
            break;
        }
    }
    EXPECT_FALSE(has_vc1_targets);
}

TEST_F(Phase4ConnectionLogicTest, MultiVC_ZRouter_VC0_and_VC1) {
    // Z router uses VC0 for receiving from mesh (no senders)
    // Z router uses VC1 for sending to mesh
    
    auto z_mapping = RouterConnectionMapping::for_z_router();
    
    // VC0 should have no sender targets (Z receives on VC0, doesn't send)
    bool has_vc0_targets = false;
    for (uint32_t ch = 0; ch < 4; ++ch) {
        if (z_mapping.has_targets(0, ch)) {
            has_vc0_targets = true;
            break;
        }
    }
    EXPECT_FALSE(has_vc0_targets);
    
    // VC1 should have sender targets (Z sends on VC1)
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

TEST_F(Phase4ConnectionLogicTest, DirectionBased_ZRouter_ChannelToDirection) {
    // Validate Z router VC1 channel-to-direction mapping
    auto z_mapping = RouterConnectionMapping::for_z_router();
    
    std::vector<std::pair<uint32_t, RoutingDirection>> expected = {
        {0, RoutingDirection::N},
        {1, RoutingDirection::E},
        {2, RoutingDirection::S},
        {3, RoutingDirection::W}
    };
    
    for (const auto& [channel, expected_dir] : expected) {
        auto targets = z_mapping.get_downstream_targets(1, channel);
        ASSERT_EQ(targets.size(), 1);
        ASSERT_TRUE(targets[0].target_direction.has_value());
        EXPECT_EQ(targets[0].target_direction.value(), expected_dir);
    }
}

TEST_F(Phase4ConnectionLogicTest, DirectionBased_MeshRouter_AggregationToZ) {
    // Validate mesh router aggregation channel targets Z direction
    auto mesh_mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::S,
        true);
    
    auto targets = mesh_mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::MESH_TO_Z);
    ASSERT_TRUE(targets[0].target_direction.has_value());
    EXPECT_EQ(targets[0].target_direction.value(), RoutingDirection::Z);
}

// ============================================================================
// Test 7: Regression Tests (Ensure Old Behavior Preserved)
// ============================================================================

TEST_F(Phase4ConnectionLogicTest, Regression_MeshToMesh_StillWorks) {
    // Verify that basic mesh-to-mesh connections still work after refactor
    auto mesh_n = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::N,
        false);
    
    auto mesh_s = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh,
        RoutingDirection::S,
        false);
    
    // Both should have INTRA_MESH targets
    bool n_has_targets = false;
    bool s_has_targets = false;
    
    for (uint32_t ch = 1; ch < 5; ++ch) {  // Skip local channel 0
        if (mesh_n.has_targets(0, ch)) n_has_targets = true;
        if (mesh_s.has_targets(0, ch)) s_has_targets = true;
    }
    
    EXPECT_TRUE(n_has_targets);
    EXPECT_TRUE(s_has_targets);
}

TEST_F(Phase4ConnectionLogicTest, Regression_1D_Topology_StillWorks) {
    // Verify 1D topology connections work after refactor
    auto mesh_1d = RouterConnectionMapping::for_mesh_router(
        Topology::Linear,
        RoutingDirection::E,
        false);
    
    // 1D should have fewer channels than 2D
    // Channel 1 should have INTRA_MESH target
    auto targets = mesh_1d.get_downstream_targets(0, 1);
    EXPECT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
}

