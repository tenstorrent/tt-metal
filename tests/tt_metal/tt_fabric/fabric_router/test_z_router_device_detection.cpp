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
    // EAST router receiver channel 0 connects to: WEST, NORTH, SOUTH
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 3) << "Receiver channel 0 should have 3 targets";

    for (const auto& target : targets) {
        EXPECT_EQ(target.type, ConnectionType::INTRA_MESH);
        EXPECT_EQ(target.target_vc, 0) << "INTRA_MESH should target VC0";
    }

    // No other receiver channels should exist
    EXPECT_TRUE(mapping.get_downstream_targets(0, 1).empty());
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_NoZ_AllDirections) {
    // Test all 4 mesh directions without Z
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

    for (auto direction : directions) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh, direction, false);

        // Receiver channel 0 should have 3 INTRA_MESH targets
        auto targets = mapping.get_downstream_targets(0, 0);
        ASSERT_EQ(targets.size(), 3) << "Direction " << static_cast<int>(direction);

        for (const auto& target : targets) {
            EXPECT_EQ(target.type, ConnectionType::INTRA_MESH);
        }

        // No other receiver channels
        EXPECT_TRUE(mapping.get_downstream_targets(0, 1).empty());
    }
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_NoZ_1D_NoExtraChannel) {
    // 1D mesh router without Z
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::E, false);

    // 1D receiver channel 0 has only 1 INTRA_MESH target
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 1);
    EXPECT_EQ(targets[0].type, ConnectionType::INTRA_MESH);

    // No other receiver channels
    EXPECT_TRUE(mapping.get_downstream_targets(0, 1).empty());
}

// ============ Mesh Router With Z Tests ============

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_HasMeshToZConnection) {
    // Mesh router WITH Z connections (device with vertical stacking)
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);

    // Receiver channel 0 should have 4 targets: 3 INTRA_MESH + 1 MESH_TO_Z
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 4);

    uint32_t intra_mesh_count = 0;
    uint32_t mesh_to_z_count = 0;

    for (const auto& target : targets) {
        if (target.type == ConnectionType::INTRA_MESH) {
            intra_mesh_count++;
            EXPECT_EQ(target.target_vc, 0) << "INTRA_MESH should target VC0";
        } else if (target.type == ConnectionType::MESH_TO_Z) {
            mesh_to_z_count++;
            EXPECT_EQ(target.target_direction.value(), RoutingDirection::Z);
            EXPECT_EQ(target.target_vc, 0) << "MESH_TO_Z should target Z router VC0";
        }
    }

    EXPECT_EQ(intra_mesh_count, 3);
    EXPECT_EQ(mesh_to_z_count, 1);
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_CorrectChannelCounts) {
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);

    // Receiver channel 0 should have 4 targets (3 mesh + 1 Z)
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 4);

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

    // No other receiver channels should exist
    EXPECT_TRUE(mapping.get_downstream_targets(0, 1).empty());
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_AllDirections) {
    // Test all 4 mesh directions with Z
    std::vector<RoutingDirection> directions = {
        RoutingDirection::N, RoutingDirection::E, RoutingDirection::S, RoutingDirection::W};

    for (auto direction : directions) {
        auto mapping = RouterConnectionMapping::for_mesh_router(
            Topology::Mesh, direction, true);

        // Receiver channel 0 should have 4 targets: 3 INTRA_MESH + 1 MESH_TO_Z
        auto targets = mapping.get_downstream_targets(0, 0);
        ASSERT_EQ(targets.size(), 4) << "Direction " << static_cast<int>(direction);

        uint32_t intra_mesh_count = 0;
        uint32_t mesh_to_z_count = 0;

        for (const auto& target : targets) {
            if (target.type == ConnectionType::INTRA_MESH) {
                intra_mesh_count++;
            } else if (target.type == ConnectionType::MESH_TO_Z) {
                mesh_to_z_count++;
                EXPECT_EQ(target.target_direction.value(), RoutingDirection::Z);
            }
        }

        EXPECT_EQ(intra_mesh_count, 3);
        EXPECT_EQ(mesh_to_z_count, 1);
    }
}

TEST_F(ZRouterDeviceDetectionTest, MeshRouter_WithZ_1D_HasZConnection) {
    // 1D mesh router with Z
    auto mapping = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::E, true);

    // 1D receiver channel 0 has 2 targets: 1 INTRA_MESH + 1 MESH_TO_Z
    auto targets = mapping.get_downstream_targets(0, 0);
    ASSERT_EQ(targets.size(), 2);

    auto mesh_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::INTRA_MESH; });
    ASSERT_NE(mesh_it, targets.end());

    auto z_it = std::find_if(targets.begin(), targets.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    ASSERT_NE(z_it, targets.end());
    EXPECT_EQ(z_it->target_direction.value(), RoutingDirection::Z);
}

// ============ Z Router Tests ============

TEST_F(ZRouterDeviceDetectionTest, ZRouter_VC1_FourDirectionalConnections) {
    auto mapping = RouterConnectionMapping::for_z_router();

    // Z router VC1 receiver channel 0 should connect to all 4 mesh directions
    std::vector<RoutingDirection> expected_directions = {
        RoutingDirection::N,
        RoutingDirection::E,
        RoutingDirection::S,
        RoutingDirection::W
    };

    auto targets = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4) << "VC1 receiver channel 0 should have 4 targets";

    // Verify all expected directions are present
    for (const auto& expected_dir : expected_directions) {
        auto it = std::find_if(targets.begin(), targets.end(), [expected_dir](const ConnectionTarget& t) {
            return t.target_direction == expected_dir;
        });
        ASSERT_NE(it, targets.end()) << "Missing direction: " << static_cast<int>(expected_dir);
        EXPECT_EQ(it->type, ConnectionType::Z_TO_MESH);
        EXPECT_EQ(it->target_vc, 1) << "Z_TO_MESH should target mesh router VC1";
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

    // All Z_TO_MESH connections from receiver channel 0 should target VC1 on mesh routers
    auto targets = mapping.get_downstream_targets(1, 0);
    ASSERT_EQ(targets.size(), 4);

    for (const auto& target : targets) {
        EXPECT_EQ(target.target_vc, 1) << "Z traffic should target mesh router VC1, not VC0";
    }
}

// ============ Comparison Tests ============

TEST_F(ZRouterDeviceDetectionTest, Comparison_MeshWithZ_vs_MeshWithoutZ) {
    auto mapping_no_z = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, false);
    auto mapping_with_z = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);

    // Both should have same INTRA_MESH connections from receiver channel 0
    auto targets_no_z = mapping_no_z.get_downstream_targets(0, 0);
    auto targets_with_z = mapping_with_z.get_downstream_targets(0, 0);

    ASSERT_EQ(targets_no_z.size(), 3);  // 3 INTRA_MESH
    ASSERT_EQ(targets_with_z.size(), 4);  // 3 INTRA_MESH + 1 MESH_TO_Z

    // Verify INTRA_MESH count
    uint32_t intra_mesh_no_z = std::count_if(targets_no_z.begin(), targets_no_z.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::INTRA_MESH; });
    uint32_t intra_mesh_with_z = std::count_if(targets_with_z.begin(), targets_with_z.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::INTRA_MESH; });

    EXPECT_EQ(intra_mesh_no_z, 3);
    EXPECT_EQ(intra_mesh_with_z, 3);

    // Only mapping_with_z should have MESH_TO_Z
    auto mesh_to_z_it = std::find_if(targets_with_z.begin(), targets_with_z.end(),
        [](const ConnectionTarget& t) { return t.type == ConnectionType::MESH_TO_Z; });
    EXPECT_NE(mesh_to_z_it, targets_with_z.end());
}

TEST_F(ZRouterDeviceDetectionTest, Comparison_2D_vs_1D_WithZ) {
    auto mapping_2d = RouterConnectionMapping::for_mesh_router(
        Topology::Mesh, RoutingDirection::E, true);
    auto mapping_1d = RouterConnectionMapping::for_mesh_router(
        Topology::Linear, RoutingDirection::E, true);

    // 2D: receiver channel 0 has 4 targets (3 mesh + 1 Z)
    auto targets_2d = mapping_2d.get_downstream_targets(0, 0);
    ASSERT_EQ(targets_2d.size(), 4);

    // 1D: receiver channel 0 has 2 targets (1 mesh + 1 Z)
    auto targets_1d = mapping_1d.get_downstream_targets(0, 0);
    ASSERT_EQ(targets_1d.size(), 2);

    // Verify no other receiver channels exist
    EXPECT_TRUE(mapping_2d.get_downstream_targets(0, 1).empty());
    EXPECT_TRUE(mapping_1d.get_downstream_targets(0, 1).empty());
}

}  // namespace tt::tt_fabric
