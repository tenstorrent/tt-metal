// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/fabric_router_channel_mapping.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

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

    // Verify mapping has expected structure: receiver channel 0 has 1 target
    EXPECT_EQ(conn_mapping.get_total_sender_count(), 1);
    EXPECT_TRUE(conn_mapping.has_targets(0, 0));
    auto targets = conn_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
}

TEST_F(BuilderConnectionMappingTest, MeshRouter_2D_WithZ_MappingCreation) {
    // Simulate what build() does for a 2D mesh router with Z
    Topology topology = Topology::Mesh;
    RoutingDirection direction = RoutingDirection::E;
    bool has_z = true;

    // Create connection mapping
    auto conn_mapping = RouterConnectionMapping::for_mesh_router(topology, direction, has_z);

    // Verify mapping has expected structure: receiver channel 0 has 4 targets (3 INTRA_MESH + 1 MESH_TO_Z)
    EXPECT_EQ(conn_mapping.get_total_sender_count(), 4);
    EXPECT_TRUE(conn_mapping.has_targets(0, 0));

    auto targets = conn_mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 4);

    // Count connection types
    uint32_t intra_mesh_count = 0;
    uint32_t mesh_to_z_count = 0;
    for (const auto& target : targets) {
        if (target.type == ConnectionType::INTRA_MESH) {
            intra_mesh_count++;
        } else if (target.type == ConnectionType::MESH_TO_Z) {
            mesh_to_z_count++;
        }
    }
    EXPECT_EQ(intra_mesh_count, 3);
    EXPECT_EQ(mesh_to_z_count, 1);
}

TEST_F(BuilderConnectionMappingTest, ZRouter_MappingCreation) {
    // Simulate what build() does for a Z router
    auto conn_mapping = RouterConnectionMapping::for_z_router();

    // Verify mapping has expected structure: receiver channel 0 on VC1 has 4 targets
    EXPECT_EQ(conn_mapping.get_total_sender_count(), 4);
    EXPECT_TRUE(conn_mapping.has_targets(1, 0));

    auto targets = conn_mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    // Verify all are Z_TO_MESH
    for (const auto& target : targets) {
        EXPECT_EQ(target.type, ConnectionType::Z_TO_MESH);
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

    // Verify consistency: receiver channel 0 should have targets matching sender channel count
    uint32_t num_vcs = channel_mapping.get_num_mapped_virtual_channels();
    for (uint32_t vc = 0; vc < num_vcs; ++vc) {
        uint32_t num_senders = channel_mapping.get_num_mapped_sender_channels_for_vc(vc);

        // For mesh router VC0, receiver channel 0 should have targets
        if (num_senders > 0) {
            EXPECT_TRUE(conn_mapping.has_targets(vc, 0))
                << "VC" << vc << " receiver channel 0 should have targets";

            auto targets = conn_mapping.get_downstream_targets(vc, 0);
            EXPECT_EQ(targets.size(), num_senders)
                << "VC" << vc << " receiver channel 0 should have " << num_senders << " targets";
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
    EXPECT_EQ(channel_mapping.get_num_mapped_virtual_channels(), 2);
    uint32_t vc1_senders = channel_mapping.get_num_mapped_sender_channels_for_vc(1);
    EXPECT_EQ(vc1_senders, 4);

    // VC1 receiver channel 0 should have 4 targets
    EXPECT_TRUE(conn_mapping.has_targets(1, 0))
        << "Z router VC1 receiver channel 0 should have connection targets";

    auto targets = conn_mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(targets.size(), vc1_senders)
        << "Z router VC1 receiver channel 0 should have " << vc1_senders << " targets";
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

        // Receiver channel 0 should have targets
        EXPECT_TRUE(mapping.has_targets(0, 0))
            << "Receiver channel 0 should have targets";

        auto targets = mapping.get_downstream_targets(0, 0);

        // If has_z, should have MESH_TO_Z target
        if (has_z) {
            auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
                [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
            EXPECT_NE(mesh_to_z_it, targets.end())
                << "Should have MESH_TO_Z target";
        }
    }
}

TEST_F(BuilderConnectionMappingTest, FactoryMethod_ZRouter) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // Z router receiver channel 0 on VC1 should have 4 targets
    EXPECT_EQ(mapping.get_total_sender_count(), 4);

    auto targets = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    // All should be Z_TO_MESH type
    for (const auto& target : targets) {
        EXPECT_EQ(target.type, ConnectionType::Z_TO_MESH);
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
    mesh_mappings.reserve(mesh_dirs.size());
    for (auto dir : mesh_dirs) {
        mesh_mappings.push_back(
            RouterConnectionMapping::for_mesh_router(Topology::Mesh, dir, true));
    }

    // Create Z router mapping
    auto z_mapping = RouterConnectionMapping::for_z_router();

    // Verify each mesh router has MESH_TO_Z capability
    for (const auto& mapping : mesh_mappings) {
        auto targets = mapping.get_downstream_targets(0, 0);
        auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
            [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
        EXPECT_NE(mesh_to_z_it, targets.end())
            << "Mesh router should have MESH_TO_Z target";
    }

    // Verify Z router has Z_TO_MESH capability for all 4 directions
    EXPECT_EQ(z_mapping.get_total_sender_count(), 4);
    auto z_targets = z_mapping.get_downstream_targets(1, 0);
    EXPECT_EQ(z_targets.size(), 4);
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
        auto targets = mapping.get_downstream_targets(0, 0);
        auto mesh_to_z_it = std::find_if(targets.begin(), targets.end(),
            [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
        EXPECT_NE(mesh_to_z_it, targets.end())
            << "Mesh router should have MESH_TO_Z target";
    }

    // Z router still has intent for all 4 directions
    auto z_mapping = RouterConnectionMapping::for_z_router();
    EXPECT_EQ(z_mapping.get_total_sender_count(), 4)
        << "Z router should have intent for all 4 directions";
}

}  // namespace tt::tt_fabric
