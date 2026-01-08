// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/test_utils/env_vars.hpp"

using namespace tt::tt_fabric;
using tt::tt_metal::distributed::MeshShape;

// =============================================================================
// Pure Unit Tests - No Fixture, Synthetic Data
// =============================================================================

TEST(FabricTopologyHelpers, Max1DHops_EmptyMeshes) {
    std::vector<MeshShape> empty;
    EXPECT_EQ(compute_max_1d_hops(empty), 0);
}

TEST(FabricTopologyHelpers, Max1DHops_SingleChip) {
    std::vector<MeshShape> single = {MeshShape{1, 1}};
    EXPECT_EQ(compute_max_1d_hops(single), 0);
}

TEST(FabricTopologyHelpers, Max1DHops_LinearMesh_1x8) {
    // T3K: 1 row, 8 columns = 7 hops (0->7)
    std::vector<MeshShape> t3k = {MeshShape{1, 8}};
    EXPECT_EQ(compute_max_1d_hops(t3k), 7);
}

TEST(FabricTopologyHelpers, Max1DHops_LinearMesh_8x1) {
    // Vertical layout: 8 rows, 1 column = 7 hops
    std::vector<MeshShape> vertical = {MeshShape{8, 1}};
    EXPECT_EQ(compute_max_1d_hops(vertical), 7);
}

TEST(FabricTopologyHelpers, Max1DHops_MultiMesh_TwoLinear) {
    // Two separate 1x4 meshes: max dimension is 4, so 3 hops
    std::vector<MeshShape> two_meshes = {MeshShape{1, 4}, MeshShape{1, 4}};
    EXPECT_EQ(compute_max_1d_hops(two_meshes), 3);
}

TEST(FabricTopologyHelpers, Max1DHops_SquareMesh_2x2) {
    // 2x2 grid: max(2,2) = 2, so 1 hop
    std::vector<MeshShape> square = {MeshShape{2, 2}};
    EXPECT_EQ(compute_max_1d_hops(square), 1);
}

TEST(FabricTopologyHelpers, Max1DHops_LargeMesh_32x4) {
    // Galaxy 1x32 (if viewed as single mesh): max(1,32) = 32, so 31 hops
    std::vector<MeshShape> galaxy_linear = {MeshShape{1, 32}};
    EXPECT_EQ(compute_max_1d_hops(galaxy_linear), 31);
}

TEST(FabricTopologyHelpers, Max1DHops_RectangularMesh_4x8) {
    // 4x8: max(4, 8) = 8, so 7 hops
    std::vector<MeshShape> rect = {MeshShape{4, 8}};
    EXPECT_EQ(compute_max_1d_hops(rect), 7);
}

// =============================================================================
// 2D Hop Tests - Pure Unit
// =============================================================================

TEST(FabricTopologyHelpers, Max2DHops_EmptyMeshes) {
    std::vector<MeshShape> empty;
    EXPECT_EQ(compute_max_2d_hops(empty), 0);
}

TEST(FabricTopologyHelpers, Max2DHops_SingleChip) {
    std::vector<MeshShape> single = {MeshShape{1, 1}};
    EXPECT_EQ(compute_max_2d_hops(single), 0);
}

TEST(FabricTopologyHelpers, Max2DHops_SquareMesh_2x2) {
    // 2x2: (2-1) + (2-1) = 1 + 1 = 2 hops
    std::vector<MeshShape> square = {MeshShape{2, 2}};
    EXPECT_EQ(compute_max_2d_hops(square), 2);
}

TEST(FabricTopologyHelpers, Max2DHops_RectangularMesh_2x4) {
    // 2x4: (2-1) + (4-1) = 1 + 3 = 4 hops
    std::vector<MeshShape> rect = {MeshShape{2, 4}};
    EXPECT_EQ(compute_max_2d_hops(rect), 4);
}

TEST(FabricTopologyHelpers, Max2DHops_Galaxy_4x8) {
    // Galaxy: 4 rows, 8 cols = (4-1) + (8-1) = 3 + 7 = 10 hops
    std::vector<MeshShape> galaxy = {MeshShape{4, 8}};
    EXPECT_EQ(compute_max_2d_hops(galaxy), 10);
}

TEST(FabricTopologyHelpers, Max2DHops_MultiMesh_FindsMax) {
    // Multiple meshes: {2x4, 8x8, 1x3}
    // Max is 8x8: (8-1) + (8-1) = 14 hops
    std::vector<MeshShape> multi = {MeshShape{2, 4}, MeshShape{8, 8}, MeshShape{1, 3}};
    EXPECT_EQ(compute_max_2d_hops(multi), 14);
}

TEST(FabricTopologyHelpers, Max2DHops_LinearMesh_1x8) {
    // Linear (T3K): 1x8 = (1-1) + (8-1) = 0 + 7 = 7 hops
    std::vector<MeshShape> t3k = {MeshShape{1, 8}};
    EXPECT_EQ(compute_max_2d_hops(t3k), 7);
}

TEST(FabricTopologyHelpers, Max2DHops_LargeMesh_32x1) {
    // Vertical 32x1: (32-1) + (1-1) = 31 + 0 = 31 hops
    std::vector<MeshShape> vertical = {MeshShape{32, 1}};
    EXPECT_EQ(compute_max_2d_hops(vertical), 31);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(FabricTopologyHelpers, Max1DHops_MaxDimension) {
    // Test with large dimensions
    std::vector<MeshShape> large = {MeshShape{1, 1000}};
    EXPECT_EQ(compute_max_1d_hops(large), 999);
}

TEST(FabricTopologyHelpers, Max2DHops_MaxDimensions) {
    std::vector<MeshShape> large = {MeshShape{100, 100}};
    EXPECT_EQ(compute_max_2d_hops(large), 198);  // (100-1) + (100-1)
}

TEST(FabricTopologyHelpers, MixedDimensions_FindsCorrectMax) {
    // Mix of 1D and 2D meshes
    std::vector<MeshShape> mixed = {MeshShape{1, 8}, MeshShape{4, 4}, MeshShape{2, 16}};
    // Max 1D: max(8, 4, 16) = 16, so 15 hops
    EXPECT_EQ(compute_max_1d_hops(mixed), 15);
    // Max 2D: max((1-1)+(8-1)=7, (4-1)+(4-1)=6, (2-1)+(16-1)=16) = 16 hops
    EXPECT_EQ(compute_max_2d_hops(mixed), 16);
}

TEST(FabricTopologyHelpers, Consistency_1DvsLinear2D) {
    // For pure 1D meshes (1xN or Nx1), 1D and 2D hop counts should match
    std::vector<MeshShape> linear_1xN = {MeshShape{1, 10}};
    EXPECT_EQ(compute_max_1d_hops(linear_1xN), compute_max_2d_hops(linear_1xN));

    std::vector<MeshShape> linear_Nx1 = {MeshShape{10, 1}};
    EXPECT_EQ(compute_max_1d_hops(linear_Nx1), compute_max_2d_hops(linear_Nx1));
}

// =============================================================================
// Mock Cluster Integration Tests (CPU-only)
// =============================================================================

class MockClusterTopologyFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Verify we're using mock cluster (required for CPU-only tests)
        auto* mock_desc = getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH");
        if (!mock_desc) {
            GTEST_SKIP() << "TT_METAL_MOCK_CLUSTER_DESC_PATH not set - skipping mock cluster tests";
        }
    }

    // Helper to extract mesh shapes from ControlPlane
    std::vector<MeshShape> get_mesh_shapes_from_control_plane() {
        auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        const auto& mesh_graph = control_plane.get_mesh_graph();

        std::vector<MeshShape> shapes;
        const auto& all_meshes = mesh_graph.get_all_mesh_ids();
        for (const auto& mesh_id : all_meshes) {
            auto mesh_shape = mesh_graph.get_mesh_shape(mesh_id);
            shapes.push_back(mesh_shape);
        }
        return shapes;
    }
};

TEST_F(MockClusterTopologyFixture, HopCalculations_WithRealTopology) {
    auto shapes = get_mesh_shapes_from_control_plane();

    // Verify hop calculations work with real data
    uint32_t max_1d = compute_max_1d_hops(shapes);
    uint32_t max_2d = compute_max_2d_hops(shapes);

    // For any valid topology, should have >= 0 hops
    EXPECT_GE(max_1d, 0);
    EXPECT_GE(max_2d, 0);

    // 2D hops should be >= 1D hops (Manhattan distance >= linear)
    EXPECT_GE(max_2d, max_1d);
}

// =============================================================================
// Known Cluster Configuration Tests
// =============================================================================

TEST_F(MockClusterTopologyFixture, KnownClusterTypes_HaveExpectedHops) {
    // Query actual cluster type from cluster descriptor
    const auto cluster_type = tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type();

    auto shapes = get_mesh_shapes_from_control_plane();

    // Expected hop values for known cluster types
    struct ExpectedHops {
        uint32_t max_1d;
        uint32_t max_2d;
        std::string description;
    };

    std::unordered_map<tt::tt_metal::ClusterType, ExpectedHops> expected_values = {
        {tt::tt_metal::ClusterType::T3K, {3, 4, "T3K: 2X4 2D mesh"}},
        {tt::tt_metal::ClusterType::GALAXY, {7, 10, "Galaxy: 4x8 2D mesh"}},
        {tt::tt_metal::ClusterType::N150, {0, 0, "N150: single chip"}},
        {tt::tt_metal::ClusterType::N300, {1, 2, "N300: 2x2 mesh"}},
        {tt::tt_metal::ClusterType::P150, {0, 0, "P150: single chip"}},
        {tt::tt_metal::ClusterType::N300_2x2, {1, 2, "N300 2x2: 2x2 mesh"}}
        // Add more cluster types as needed
    };

    auto it = expected_values.find(cluster_type);
    if (it != expected_values.end()) {
        const auto& expected = it->second;
        uint32_t actual_1d = compute_max_1d_hops(shapes);
        uint32_t actual_2d = compute_max_2d_hops(shapes);

        log_debug(
            tt::LogTest,
            "Testing {} - Expected: 1D={}, 2D={} | Actual: 1D={}, 2D={}",
            expected.description,
            expected.max_1d,
            expected.max_2d,
            actual_1d,
            actual_2d);

        EXPECT_EQ(actual_1d, expected.max_1d) << "Incorrect 1D hop count for " << expected.description;
        EXPECT_EQ(actual_2d, expected.max_2d) << "Incorrect 2D hop count for " << expected.description;
    } else {
        GTEST_SKIP() << "No expected hop values defined for cluster type: " << enchantum::to_string(cluster_type);
    }
}
