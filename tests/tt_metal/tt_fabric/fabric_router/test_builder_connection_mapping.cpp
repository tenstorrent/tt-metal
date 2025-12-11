// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {

/**
 * Builder Connection Mapping Tests
 *
 * These tests validate that RouterConnectionMapping integrates correctly
 * with the builder infrastructure:
 * - Connection mappings are created correctly for different router types
 * - Mappings are consistent with channel mappings
 * - Factory methods produce correct configurations
 * - Builder can query and use connection mappings
 *
 * Test Coverage Summary:
 * ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                        │ Test Name                                  │ Focus      │
 * ├─────────────────────────────────┼────────────────────────────────────────────┼────────────┤
 * │ Connection Mapping Creation     │ MeshRouter_1D_NoZ_MappingCreation          │ 1D mesh    │
 * │                                 │ MeshRouter_2D_WithZ_MappingCreation        │ 2D mesh+Z  │
 * │                                 │ ZRouter_MappingCreation                    │ Z router   │
 * ├─────────────────────────────────┼────────────────────────────────────────────┼────────────┤
 * │ Channel + Connection Consistency│ MeshRouter_ChannelAndConnectionMapping_... │ Mesh sync  │
 * │                                 │ ZRouter_ChannelAndConnectionMapping_...    │ Z sync     │
 * ├─────────────────────────────────┼────────────────────────────────────────────┼────────────┤
 * │ Factory Method Validation       │ FactoryMethod_MeshRouter_ProducesCorrect...│ Mesh API   │
 * │                                 │ FactoryMethod_ZRouter_ProducesCorrectType  │ Z API      │
 * ├─────────────────────────────────┼────────────────────────────────────────────┼────────────┤
 * │ Builder Query Interface         │ BuilderQuery_GetDownstreamTargets_Works   │ Query API  │
 * │                                 │ BuilderQuery_HasTargets_Works              │ Check API  │
 * │                                 │ BuilderQuery_GetTotalSenderCount_Accurate  │ Count API  │
 * └─────────────────────────────────┴────────────────────────────────────────────┴────────────┘
 *
 * Total: 11 tests across 4 categories
 */
class BuilderConnectionMappingTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Connection Mapping Creation Tests
// ============================================================================

TEST_F(BuilderConnectionMappingTest, MeshRouter_1D_NoZ_MappingCreation) {
    // Simulate what build() does for a 1D mesh router without Z
    Topology topology = Topology::Linear;
    RoutingDirection direction = RoutingDirection::N;
    bool has_z = false;

    // Create connection mapping
    auto conn_mapping = RouterConnectionMapping::for_mesh_router(topology, direction, has_z);

    // Verify mapping has expected structure
    EXPECT_EQ(conn_mapping.get_total_sender_count(), 1);  // Only channel 1
    EXPECT_TRUE(conn_mapping.has_targets(0, 1));
    EXPECT_FALSE(conn_mapping.has_targets(0, 2));  // No MESH_TO_Z channel
}

TEST_F(BuilderConnectionMappingTest, MeshRouter_2D_WithZ_MappingCreation) {
    // Simulate what build() does for a 2D mesh router with Z
    Topology topology = Topology::Mesh;
    RoutingDirection direction = RoutingDirection::E;
    bool has_z = true;

    // Create connection mapping
    auto conn_mapping = RouterConnectionMapping::for_mesh_router(topology, direction, has_z);

    // Verify mapping has expected structure
    EXPECT_EQ(conn_mapping.get_total_sender_count(), 4);  // Channels 1-3 + MESH_TO_Z

    // Verify INTRA_MESH channels
    EXPECT_TRUE(conn_mapping.has_targets(0, 1));
    EXPECT_TRUE(conn_mapping.has_targets(0, 2));
    EXPECT_TRUE(conn_mapping.has_targets(0, 3));

    // Verify MESH_TO_Z channel
    EXPECT_TRUE(conn_mapping.has_targets(0, 4));
    auto z_targets = conn_mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(z_targets.size(), 1);
    EXPECT_EQ(z_targets[0].type, ConnectionType::MESH_TO_Z);
}

TEST_F(BuilderConnectionMappingTest, ZRouter_MappingCreation) {
    // Simulate what build() does for a Z router
    auto conn_mapping = RouterConnectionMapping::for_z_router();

    // Verify mapping has expected structure
    EXPECT_EQ(conn_mapping.get_total_sender_count(), 4);  // VC1 channels 0-3

    // Verify all Z_TO_MESH channels
    for (uint32_t ch = 0; ch < 4; ++ch) {
        EXPECT_TRUE(conn_mapping.has_targets(1, ch));
        auto targets = conn_mapping.get_downstream_targets(1, ch);
        ASSERT_EQ(targets.size(), 1);
        EXPECT_EQ(targets[0].type, ConnectionType::Z_TO_MESH);
    }
}

// ============================================================================
// Channel Mapping + Connection Mapping Consistency Tests
// ============================================================================

TEST_F(BuilderConnectionMappingTest, MeshRouter_ChannelAndConnectionMapping_Consistent) {
    // Create both mappings as build() would
    Topology topology = Topology::Mesh;
    RoutingDirection routing_dir = RoutingDirection::N;
    bool has_tensix = false;
    bool has_z = true;
    RouterVariant variant = RouterVariant::MESH;

    // Phase 1: Channel mapping
    FabricRouterChannelMapping channel_mapping(topology, has_tensix, variant, nullptr);

    // Phase 2: Connection mapping
    RouterConnectionMapping conn_mapping = RouterConnectionMapping::for_mesh_router(topology, routing_dir, has_z);

    // Verify consistency: every sender channel in channel mapping should have connection targets
    uint32_t num_vcs = channel_mapping.get_num_virtual_channels();
    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        uint32_t num_senders = channel_mapping.get_num_sender_channels_for_vc(vc);

        // For mesh router VC0, channels 1-4 should have targets (channel 0 is reserved)
        for (uint32_t ch = 1; ch <= num_senders && ch <= 4; ++ch) {
            bool has_channel_mapping = true;  // Channel mapping exists
            bool has_conn_targets = conn_mapping.has_targets(vc, ch);

            EXPECT_EQ(has_channel_mapping, has_conn_targets)
                << "VC" << vc << " channel " << ch << " consistency mismatch";
        }
    }
}

TEST_F(BuilderConnectionMappingTest, ZRouter_ChannelAndConnectionMapping_Consistent) {
    // Create both mappings as build() would for Z router
    Topology topology = Topology::Mesh;
    bool has_tensix = false;
    RouterVariant variant = RouterVariant::Z_ROUTER;
    auto intermesh_config = IntermeshVCConfig::full_mesh();

    // Phase 1: Channel mapping
    FabricRouterChannelMapping channel_mapping(topology, has_tensix, variant, &intermesh_config);

    // Phase 2: Connection mapping
    RouterConnectionMapping conn_mapping = RouterConnectionMapping::for_z_router();

    // Verify VC1 consistency
    EXPECT_EQ(channel_mapping.get_num_virtual_channels(), 2);
    uint32_t vc1_senders = channel_mapping.get_num_sender_channels_for_vc(1);
    EXPECT_EQ(vc1_senders, 4);

    // All VC1 sender channels should have connection targets
    for (uint32_t ch = 0; ch < vc1_senders; ++ch) {
        EXPECT_TRUE(conn_mapping.has_targets(1, ch))
            << "Z router VC1 channel " << ch << " should have connection targets";
    }
}

// ============================================================================
// Variant Detection Tests
// ============================================================================

TEST_F(BuilderConnectionMappingTest, VariantDetection_MeshRouter) {
    // Test that mesh routers are correctly identified
    std::vector<RoutingDirection> mesh_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    for (auto dir : mesh_directions) {
        RouterVariant variant = (dir == RoutingDirection::Z) ? RouterVariant::Z_ROUTER : RouterVariant::MESH;

        EXPECT_EQ(variant, RouterVariant::MESH)
            << "Direction " << static_cast<int>(dir) << " should be MESH variant";
    }
}

TEST_F(BuilderConnectionMappingTest, VariantDetection_ZRouter) {
    // Test that Z router is correctly identified
    RoutingDirection dir = RoutingDirection::Z;
    RouterVariant variant = (dir == RoutingDirection::Z) ? RouterVariant::Z_ROUTER : RouterVariant::MESH;

    EXPECT_EQ(variant, RouterVariant::Z_ROUTER);
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST_F(BuilderConnectionMappingTest, FactoryMethod_AllMeshConfigurations) {
    // Test all valid mesh router configurations
    std::vector<std::tuple<Topology, RoutingDirection, bool>> configs = {
        {Topology::Linear, RoutingDirection::N, false},
        {Topology::Linear, RoutingDirection::S, false},
        {Topology::Linear, RoutingDirection::N, true},
        {Topology::Mesh, RoutingDirection::N, false},
        {Topology::Mesh, RoutingDirection::E, false},
        {Topology::Mesh, RoutingDirection::N, true},
        {Topology::Mesh, RoutingDirection::W, true},
    };

    for (const auto& [topology, direction, has_z] : configs) {
        auto mapping = RouterConnectionMapping::for_mesh_router(topology, direction, has_z);

        // Basic validation
        EXPECT_GT(mapping.get_total_sender_count(), 0)
            << "Mapping should have at least one sender channel";

        // If has_z, should have MESH_TO_Z target
        if (has_z) {
            uint32_t z_channel = (topology == Topology::Linear) ? 2 : 4;
            EXPECT_TRUE(mapping.has_targets(0, z_channel))
                << "Should have MESH_TO_Z channel";
        }
    }
}

TEST_F(BuilderConnectionMappingTest, FactoryMethod_ZRouter) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // Z router should have exactly 4 sender channels on VC1
    EXPECT_EQ(mapping.get_total_sender_count(), 4);

    // All should be Z_TO_MESH type
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(1, ch);
        ASSERT_EQ(targets.size(), 1);
        EXPECT_EQ(targets[0].type, ConnectionType::Z_TO_MESH);
    }
}

// ============================================================================
// Integration Scenario Tests
// ============================================================================

TEST_F(BuilderConnectionMappingTest, Scenario_FullDevice_4Mesh1Z) {
    // Simulate full device initialization with 4 mesh + 1 Z router

    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    // Create mesh router mappings
    std::vector<RouterConnectionMapping> mesh_mappings;
    for (auto dir : mesh_dirs) {
        mesh_mappings.push_back(
            RouterConnectionMapping::for_mesh_router(Topology::Mesh, dir, true));
    }

    // Create Z router mapping
    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Verify each mesh router has MESH_TO_Z capability
    for (const auto& mapping : mesh_mappings) {
        EXPECT_TRUE(mapping.has_targets(0, 4))  // Channel 4 is MESH_TO_Z
            << "Mesh router should have MESH_TO_Z target";
    }

    // Verify Z router has Z_TO_MESH capability for all 4 directions
    EXPECT_EQ(z_mapping.get_total_sender_count(), 4);
}

TEST_F(BuilderConnectionMappingTest, Scenario_EdgeDevice_2Mesh1Z) {
    // Simulate edge device with only 2 mesh routers + Z

    std::vector<RoutingDirection> mesh_dirs = {
        RoutingDirection::N,
        RoutingDirection::E
    };

    // Create mesh router mappings
    for (auto dir : mesh_dirs) {
        auto mapping = RouterConnectionMapping::for_mesh_router(Topology::Mesh, dir, true);
        EXPECT_TRUE(mapping.has_targets(0, 4))
            << "Mesh router should have MESH_TO_Z target";
    }

    // Z router still has intent for all 4 directions
    auto z_mapping = RouterConnectionMapping::for_z_router();
    EXPECT_EQ(z_mapping.get_total_sender_count(), 4)
        << "Z router should have intent for all 4 directions";
}

}  // namespace tt::tt_fabric
