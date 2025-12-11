// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/builder/router_connection_mapping.hpp"
#include "tt_metal/fabric/builder/connection_registry.hpp"
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace tt::tt_fabric {

/**
 * Test fixture for Z router device detection and connection mapping
 *
 * These tests validate that:
 * - Mesh routers without Z create only INTRA_MESH connections
 * - Mesh routers with Z create INTRA_MESH + MESH_TO_Z connections
 * - Z routers create Z_TO_MESH connections on VC1
 * - Connection targets have correct VCs and directions
 *
 * All tests are pure unit tests - no devices required.
 *
 * Test Coverage Summary:
 * ┌──────────────────────────────────────────────────────────────────────────────────────┐
 * │ Category                    │ Test Name                                │ Focus        │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mesh Router Without Z       │ MeshRouter_NoZ_OnlyIntraMeshConnections  │ Basic mesh   │
 * │                             │ MeshRouter_NoZ_AllDirections             │ All dirs     │
 * │                             │ MeshRouter_NoZ_1D_NoExtraChannel         │ 1D topology  │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Mesh Router With Z          │ MeshRouter_WithZ_HasMeshToZConnection    │ Z connection │
 * │                             │ MeshRouter_WithZ_AllDirections           │ All dirs+Z   │
 * │                             │ MeshRouter_WithZ_TargetsZDirection       │ Z targeting  │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Z Router                    │ ZRouter_VC1_HasZToMeshConnections        │ VC1 senders  │
 * │                             │ ZRouter_VC0_NoSenderConnections          │ VC0 receivers│
 * │                             │ ZRouter_AllDirections_TargetMeshVC1      │ All dirs     │
 * ├─────────────────────────────┼──────────────────────────────────────────┼──────────────┤
 * │ Comparison Tests            │ Compare_MeshWithZ_vs_MeshWithoutZ        │ Diff analysis│
 * │                             │ Compare_ZRouter_vs_MeshRouter            │ Type diff    │
 * │                             │ Compare_AllThreeConfigurations           │ Full matrix  │
 * └─────────────────────────────┴──────────────────────────────────────────┴──────────────┘
 *
 * Total: 12 tests across 4 categories
 */
class ZRouterDeviceDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============ Mesh Router Without Z Tests ============

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_NoZ_OnlyIntraMeshConnections) {
    // Mesh router without Z connections (typical mesh router)
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, false);

    // Verify only INTRA_MESH connections exist on VC0
    // EAST router connects to: WEST (ch1), NORTH (ch2), SOUTH (ch3)
    for (uint32_t ch = 1; ch <= 3; ++ch) {
        auto targets = mapping.get_downstream_targets(0, ch);
        ASSERT_FALSE(targets.empty()) << "Channel " << ch << " should have targets";
        EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
        EXPECT_EQ(targets[0].target_vc, 0) << "INTRA_MESH should target VC0";
    }

    // Channel 4 should not exist (no Z connection)
    auto ch4_targets = mapping.get_downstream_targets(0, 4);
    EXPECT_TRUE(ch4_targets.empty()) << "Channel 4 should not exist without Z";
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_NoZ_AllDirections) {
    // Test all 4 mesh directions without Z
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

    for (auto direction : directions) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh, direction, false);

        // Each direction should have 3 INTRA_MESH connections
        for (uint32_t ch = 1; ch <= 3; ++ch) {
            auto targets = mapping.get_downstream_targets(0, ch);
            ASSERT_FALSE(targets.empty()) << "Direction " << static_cast<int>(direction) << " channel " << ch;
            EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
        }

        // No Z connection
        EXPECT_TRUE(mapping.get_downstream_targets(0, 4).empty());
    }
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_NoZ_1D_NoExtraChannel) {
    // 1D mesh router without Z
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::E, false);

    // 1D has only 1 INTRA_MESH connection
    auto targets = mapping.get_downstream_targets(0, 1);
    ASSERT_FALSE(targets.empty());
    EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);

    // No Z connection in 1D either
    EXPECT_TRUE(mapping.get_downstream_targets(0, 2).empty());
}

// ============ Mesh Router With Z Tests ============

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_HasMeshToZConnection) {
    // Mesh router WITH Z connections (device with vertical stacking)
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);

    // Verify INTRA_MESH connections still exist (channels 1-3)
    for (uint32_t ch = 1; ch <= 3; ++ch) {
        auto targets = mapping.get_downstream_targets(0, ch);
        ASSERT_FALSE(targets.empty()) << "Channel " << ch << " should have targets";
        EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
        EXPECT_EQ(targets[0].target_vc, 0) << "INTRA_MESH should target VC0";
    }

    // Channel 4 should have MESH_TO_Z connection
    auto z_targets = mapping.get_downstream_targets(0, 4);
    ASSERT_EQ(z_targets.size(), 1) << "Channel 4 should have exactly one Z target";
    EXPECT_EQ(z_targets[0].type, ConnectionType::MESH_TO_Z);
    EXPECT_EQ(z_targets[0].target_direction.value(), RoutingDirection::Z);
    EXPECT_EQ(z_targets[0].target_vc, 0) << "MESH_TO_Z should target Z router VC0";
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_CorrectChannelCounts) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);

    // Verify we have targets for channels 1, 2, 3 (mesh) and 4 (Z)
    EXPECT_FALSE(mapping.get_downstream_targets(0, 1).empty()) << "Channel 1 (mesh)";
    EXPECT_FALSE(mapping.get_downstream_targets(0, 2).empty()) << "Channel 2 (mesh)";
    EXPECT_FALSE(mapping.get_downstream_targets(0, 3).empty()) << "Channel 3 (mesh)";
    EXPECT_FALSE(mapping.get_downstream_targets(0, 4).empty()) << "Channel 4 (Z)";

    // Channel 5 should not exist
    EXPECT_TRUE(mapping.get_downstream_targets(0, 5).empty()) << "Channel 5 should not exist";
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_AllDirections) {
    // Test all 4 mesh directions with Z
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

    for (auto direction : directions) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh, direction, true);

        // Each direction should have 3 INTRA_MESH + 1 MESH_TO_Z
        for (uint32_t ch = 1; ch <= 3; ++ch) {
            auto targets = mapping.get_downstream_targets(0, ch);
            ASSERT_FALSE(targets.empty()) << "Direction " << static_cast<int>(direction) << " channel " << ch;
            EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);
        }

        // Z connection on channel 4
        auto z_targets = mapping.get_downstream_targets(0, 4);
        ASSERT_FALSE(z_targets.empty()) << "Direction " << static_cast<int>(direction) << " should have Z";
        EXPECT_EQ(z_targets[0].type, ConnectionType::MESH_TO_Z);
        EXPECT_EQ(z_targets[0].target_direction.value(), RoutingDirection::Z);
    }
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_1D_HasZConnection) {
    // 1D mesh router with Z
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::E, true);

    // 1D has 1 INTRA_MESH connection
    auto mesh_targets = mapping.get_downstream_targets(0, 1);
    ASSERT_FALSE(mesh_targets.empty());
    EXPECT_EQ(mesh_targets[0].type, ConnectionType::INTRA_MESH);

    // Z connection on channel 2 (after the 1 mesh channel)
    auto z_targets = mapping.get_downstream_targets(0, 2);
    ASSERT_FALSE(z_targets.empty());
    EXPECT_EQ(z_targets[0].type, ConnectionType::MESH_TO_Z);
    EXPECT_EQ(z_targets[0].target_direction.value(), RoutingDirection::Z);
}

// ============ Z Router Tests ============

TEST_F(ZRouterDeviceDetectionTest, ZRouter_VC1_FourDirectionalConnections) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // Z router VC1 should connect to all 4 mesh directions
    std::vector<RoutingDirection> expected_directions = {
        RoutingDirection::N,  // Sender channel 0
        RoutingDirection::E,  // Sender channel 1
        RoutingDirection::S,  // Sender channel 2
        RoutingDirection::W   // Sender channel 3
    };

    for (size_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(1, ch);
        ASSERT_EQ(targets.size(), 1) << "VC1 channel " << ch << " should have one target";
        EXPECT_EQ(targets[0].type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(targets[0].target_vc, 1) << "Z_TO_MESH should target mesh router VC1";
        EXPECT_EQ(targets[0].target_direction.value(), expected_directions[ch]);
    }
}

TEST_F(ZRouterDeviceDetectionTest, ZRouter_VC0_NoConnections) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // Z router VC0 is currently unused/reserved
    // Check that no connections are defined on VC0
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(0, ch);
        EXPECT_TRUE(targets.empty()) << "Z router VC0 channel " << ch << " should have no connections";
    }
}

TEST_F(ZRouterDeviceDetectionTest, ZRouter_VC1_CorrectTargetVCs) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // All Z_TO_MESH connections should target VC1 on mesh routers
    for (uint32_t ch = 0; ch < 4; ++ch) {
        auto targets = mapping.get_downstream_targets(1, ch);
        ASSERT_FALSE(targets.empty());
        EXPECT_EQ(targets[0].target_vc, 1) << "Z traffic should target mesh router VC1, not VC0";
    }
}

// ============ Comparison Tests ============

TEST_F(ZRouterDeviceDetectionTest, Comparison_MeshWithZ_vs_MeshWithoutZ) {
    auto mapping_no_z = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, false);
    auto mapping_with_z = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);

    // Both should have same INTRA_MESH connections (channels 1-3)
    for (uint32_t ch = 1; ch <= 3; ++ch) {
        auto targets_no_z = mapping_no_z.get_downstream_targets(0, ch);
        auto targets_with_z = mapping_with_z.get_downstream_targets(0, ch);

        ASSERT_FALSE(targets_no_z.empty());
        ASSERT_FALSE(targets_with_z.empty());
        EXPECT_EQ(targets_no_z[0].type, targets_with_z[0].type);
    }

    // Only mapping_with_z should have channel 4
    EXPECT_TRUE(mapping_no_z.get_downstream_targets(0, 4).empty());
    EXPECT_FALSE(mapping_with_z.get_downstream_targets(0, 4).empty());
}

TEST_F(ZRouterDeviceDetectionTest, Comparison_2D_vs_1D_WithZ) {
    auto mapping_2d = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);
    auto mapping_1d = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::E, true);

    // 2D: 3 mesh + 1 Z = 4 channels
    EXPECT_FALSE(mapping_2d.get_downstream_targets(0, 1).empty());
    EXPECT_FALSE(mapping_2d.get_downstream_targets(0, 2).empty());
    EXPECT_FALSE(mapping_2d.get_downstream_targets(0, 3).empty());
    EXPECT_FALSE(mapping_2d.get_downstream_targets(0, 4).empty());

    // 1D: 1 mesh + 1 Z = 2 channels
    EXPECT_FALSE(mapping_1d.get_downstream_targets(0, 1).empty());
    EXPECT_FALSE(mapping_1d.get_downstream_targets(0, 2).empty());
    EXPECT_TRUE(mapping_1d.get_downstream_targets(0, 3).empty());
}

}  // namespace tt::tt_fabric
