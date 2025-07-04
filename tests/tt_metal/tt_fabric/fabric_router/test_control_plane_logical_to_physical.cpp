// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
// Now include the actual ControlPlane header with friend declaration
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>

#include "fabric_fixture.hpp"

namespace tt::tt_fabric {
// Unified Configurable Mock ControlPlane class for testing get_mesh_physical_chip_ids
class TestingMockControlPlane : public ControlPlane {
private:
    // Note: This mesh graph descriptor doesn't really matter because functions that are used are overridden by below,
    //       but we need to provide a valid path to the constructor.
    static std::string get_default_mesh_graph_desc_path() {
        const std::filesystem::path mesh_graph_desc_path =
            std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
            "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
        return mesh_graph_desc_path.string();
    }

    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data_;
    std::set<chip_id_t> user_exposed_chip_ids_;
    tt::tt_metal::distributed::MeshContainer<chip_id_t> mesh_container_;

public:
    TestingMockControlPlane(
        const std::unordered_map<chip_id_t, std::vector<chip_id_t>>& adjacency_map,
        const tt::tt_metal::distributed::MeshShape& mesh_shape,
        const std::set<chip_id_t>& user_chip_ids = {}) :
        ControlPlane(get_default_mesh_graph_desc_path()),
        adjacency_data_(adjacency_map),
        user_exposed_chip_ids_(user_chip_ids),
        mesh_container_(mesh_shape, [&] {
            std::vector<chip_id_t> chip_ids;
            chip_ids.reserve(mesh_shape.mesh_size());
            for (size_t i = 0; i < mesh_shape.mesh_size(); ++i) {
                chip_ids.push_back(static_cast<chip_id_t>(i));
            }
            return chip_ids;
        }()) {}

    std::vector<chip_id_t> get_adjacent_chips(chip_id_t chip_id, std::uint32_t num_ports_per_side) const override {
        auto it = adjacency_data_.find(chip_id);
        if (it != adjacency_data_.end()) {
            return it->second;
        }
        return {};
    }

    std::set<chip_id_t> get_user_exposed_chip_ids() const override {
        if (!user_exposed_chip_ids_.empty()) {
            return user_exposed_chip_ids_;
        }
        return {};
    }

    std::vector<chip_id_t> get_mesh_physical_chip_ids(
        const tt::tt_metal::distributed::MeshContainer<chip_id_t>& mesh_container,
        std::optional<chip_id_t> starting_physical_chip_id = std::nullopt) const override {
        return ControlPlane::get_mesh_physical_chip_ids(mesh_container, starting_physical_chip_id);
    }

    // Convenience method to get mesh physical chip IDs using the internal mesh container
    std::vector<chip_id_t> get_mesh_physical_chip_ids() const { return get_mesh_physical_chip_ids(mesh_container_); }

    // Getter for the internal mesh container
    const tt::tt_metal::distributed::MeshContainer<chip_id_t>& get_mesh_container() const { return mesh_container_; }
};
}  // namespace tt::tt_fabric

namespace tt::tt_fabric::fabric_router_tests {

// Helper function to verify physical chip ID mapping
void verify_physical_chip_ids(
    const std::vector<chip_id_t>& physical_chip_ids,
    size_t expected_size,
    const std::vector<std::pair<size_t, chip_id_t>>& expected_mappings = {}) {
    // Basic size verification
    EXPECT_EQ(physical_chip_ids.size(), expected_size);

    // Verify first chip is mapped to 0
    EXPECT_EQ(physical_chip_ids[0], 0);

    // Verify no unmapped chips (no -1 values)
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }

    // Verify no duplicate chip IDs
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";

    // Verify specific mappings if provided
    for (const auto& [index, expected_chip_id] : expected_mappings) {
        EXPECT_EQ(physical_chip_ids[index], expected_chip_id)
            << "Chip at index " << index << " should be " << expected_chip_id;
    }
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsWithConfigurableMock) {
    // Test 3x3 square mesh shape
    // Shape: 0-1-2
    //        | | |
    //        3-4-5
    //        | | |
    //        6-7-8
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 3}},
        {1, {0, 2, 4}},
        {2, {1, 5}},
        {3, {0, 4, 6}},
        {4, {1, 3, 5, 7}},
        {5, {2, 4, 8}},
        {6, {3, 7}},
        {7, {4, 6, 8}},
        {8, {5, 7}}};
    TestingMockControlPlane mock_cp(
        adjacency_data, tt::tt_metal::distributed::MeshShape(3, 3), {0, 1, 2, 3, 4, 5, 6, 7, 8});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    // Verify all chip mappings for 3x3 mesh
    verify_physical_chip_ids(
        physical_chip_ids,
        9,
        {
            {0, 0},
            {1, 1},
            {2, 2},  // top row
            {3, 3},
            {4, 4},
            {5, 5},  // middle row
            {6, 6},
            {7, 7},
            {8, 8}  // bottom row
        });
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds1x1Mesh) {
    // Test 1x1 mesh shape (single chip)
    // Shape: 0
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {{0, {}}};
    TestingMockControlPlane mock_cp(adjacency_data, tt::tt_metal::distributed::MeshShape(1, 1), {0});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    verify_physical_chip_ids(physical_chip_ids, 1, {{0, 0}});
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds3x2RectangleMesh) {
    // Test 3x2 rectangle mesh shape
    // Shape: 0-1
    //        | |
    //        2-3
    //        | |
    //        4-5
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 2}}, {1, {0, 3}}, {2, {0, 3, 4}}, {3, {1, 2, 5}}, {4, {2, 5}}, {5, {3, 4}}};
    TestingMockControlPlane mock_cp(adjacency_data, tt::tt_metal::distributed::MeshShape(3, 2), {0, 1, 2, 3, 4, 5});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    // Verify all chip mappings for 3x2 mesh
    verify_physical_chip_ids(
        physical_chip_ids,
        6,
        {
            {0, 0},
            {1, 1},  // top row
            {2, 2},
            {3, 3},  // middle row
            {4, 4},
            {5, 5}  // bottom row
        });
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds4x8LargeMesh) {
    // Test 4x8 large rectangular mesh shape
    // Shape: 0-1-2-3-4-5-6-7
    //        | | | | | | | |
    //        8-9-A-B-C-D-E-F
    //        | | | | | | | |
    //        G-H-I-J-K-L-M-N
    //        | | | | | | | |
    //        O-P-Q-R-S-T-U-V
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 8}},
        {1, {0, 2, 9}},
        {2, {1, 3, 10}},
        {3, {2, 4, 11}},
        {4, {3, 5, 12}},
        {5, {4, 6, 13}},
        {6, {5, 7, 14}},
        {7, {6, 15}},
        {8, {0, 9, 16}},
        {9, {1, 8, 10, 17}},
        {10, {2, 9, 11, 18}},
        {11, {3, 10, 12, 19}},
        {12, {4, 11, 13, 20}},
        {13, {5, 12, 14, 21}},
        {14, {6, 13, 15, 22}},
        {15, {7, 14, 23}},
        {16, {8, 17, 24}},
        {17, {9, 16, 18, 25}},
        {18, {10, 17, 19, 26}},
        {19, {11, 18, 20, 27}},
        {20, {12, 19, 21, 28}},
        {21, {13, 20, 22, 29}},
        {22, {14, 21, 23, 30}},
        {23, {15, 22, 31}},
        {24, {16, 25}},
        {25, {17, 24, 26}},
        {26, {18, 25, 27}},
        {27, {19, 26, 28}},
        {28, {20, 27, 29}},
        {29, {21, 28, 30}},
        {30, {22, 29, 31}},
        {31, {23, 30}}};
    TestingMockControlPlane mock_cp(
        adjacency_data, tt::tt_metal::distributed::MeshShape(4, 8), {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                                     22, 23, 24, 25, 26, 27, 28, 29, 30, 31});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    // Verify key chip mappings for 4x8 mesh (not all 32 chips for brevity)
    verify_physical_chip_ids(
        physical_chip_ids,
        32,
        {
            {0, 0},
            {7, 7},
            {24, 24},
            {31, 31},  // corners
            {1, 1},
            {8, 8},
            {15, 15},
            {25, 25},  // edges
            {9, 9},
            {17, 17},
            {26, 26}  // interior
        });
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsOverlappingConnections) {
    // Test 3x3 mesh with overlapping connections (multiple paths between chips)
    // Shape: 0-1-2
    //        |x|x|
    //        3-4-5
    //        |x|x|
    //        6-7-8
    // Overlapping connections: 0-4, 1-3, 2-5, 3-7, 4-5, 5-7, 6-7, 7-8
    // This creates a topology where some chips have more than 4 neighbors,
    // which breaks the corner detection logic. This test demonstrates that
    // the current algorithm doesn't handle irregular topologies well.
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 3, 4}},        // Extra connection to 4
        {1, {0, 2, 3, 4, 5}},  // Extra connections to 3 and 5
        {2, {1, 5}},
        {3, {0, 1, 4, 6, 7}},  // Extra connections from 1 and to 7
        {4, {0, 1, 3, 5, 7}},  // Extra connections from 0 and to 7
        {5, {1, 2, 4, 7, 8}},  // Extra connections from 1 and to 7
        {6, {3, 7}},
        {7, {3, 4, 5, 6, 8}},  // Extra connections from 3, 4, 5
        {8, {5, 7}}};

    TestingMockControlPlane mock_cp(
        adjacency_data, tt::tt_metal::distributed::MeshShape(3, 3), {0, 1, 2, 3, 4, 5, 6, 7, 8});

    // This test should fail because the topology has irregular connectivity
    // that doesn't match the expected 4-corner pattern for 2D meshes
    // The error message will now include ASCII visualization of the topology
    EXPECT_THROW(
        { auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(); }, std::exception)
        << "Expected exception due to irregular topology with overlapping connections. "
        << "The error message should now include ASCII visualization of the mesh topology.";
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsMissingConnection) {
    // Test 3x3 mesh with a critical missing connection
    // Shape: 0-1-2
    //        | | |
    //        3-4-5
    //        |   |
    //        6-7-8
    // Missing connection: 4-7 (critical vertical connection)
    // This creates a topology where chip 4 has only 3 neighbors instead of 4,
    // and chip 7 has only 2 neighbors instead of 3, which changes the corner
    // detection pattern.
    //
    // When this test fails, the error message will now include:
    // - ASCII visualization of the actual topology
    // - Expected vs actual corner count
    // - Adjacency matrix showing all connections
    // - Grid visualization attempt
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 3}},
        {1, {0, 2, 4}},
        {2, {1, 5}},
        {3, {0, 4, 6}},
        {4, {1, 3, 5}},  // Missing connection to 7
        {5, {2, 4, 8}},
        {6, {3, 7}},
        {7, {6, 8}},  // Missing connection from 4
        {8, {5, 7}}};
    TestingMockControlPlane mock_cp(
        adjacency_data, tt::tt_metal::distributed::MeshShape(3, 3), {0, 1, 2, 3, 4, 5, 6, 7, 8});

    // This test should fail because the missing connection creates an irregular
    // topology that doesn't match the expected 4-corner pattern for 2D meshes
    EXPECT_THROW(
        { auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(); }, std::exception)
        << "Expected exception due to irregular topology with missing connection";
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsMissingChip) {
    // Test 3x3 mesh with one chip completely missing from connections
    // Shape: 0-1-2
    //        |   |
    //        3   5  (chip 4 is missing)
    //        |   |
    //        6-7-8
    // Missing chip: 4 (no connections to or from this chip)
    // All connections involving chip 4 are removed from the topology
    // This creates a topology where the remaining chips have irregular
    // connectivity patterns that don't match the expected 4-corner pattern.
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 3}},  // No connection to 4
        {1, {0, 2}},  // No connection to 4
        {2, {1, 5}},  // No connection to 4
        {3, {0, 6}},  // No connection to 4
        {5, {2, 8}},  // No connection from 4
        {6, {3, 7}},  // No connection from 4
        {7, {6, 8}},  // No connection from 4
        {8, {5, 7}}   // No connection from 4
        // Note: chip 4 is completely missing from adjacency_data
    };
    TestingMockControlPlane mock_cp(
        adjacency_data,
        tt::tt_metal::distributed::MeshShape(3, 3),
        {0, 1, 2, 3, 5, 6, 7, 8});  // 4 not in user_chip_ids

    // This test should fail because the missing chip creates an irregular
    // topology that doesn't match the expected 4-corner pattern for 2D meshes
    EXPECT_THROW(
        { auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(); }, std::exception)
        << "Expected exception due to irregular topology with missing chip";
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds4x1Mesh) {
    // Test 4x1 1D mesh shape
    // Shape: 0
    //        |
    //        1
    //        |
    //        2
    //        |
    //        3
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1}}, {1, {0, 2}}, {2, {1, 3}}, {3, {2}}};
    TestingMockControlPlane mock_cp(adjacency_data, tt::tt_metal::distributed::MeshShape(4, 1), {0, 1, 2, 3});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    // Verify all chip mappings for 4x1 mesh
    verify_physical_chip_ids(
        physical_chip_ids,
        4,
        {
            {0, 0}, {1, 1}, {2, 2}, {3, 3}  // vertical line
        });
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds1x8Mesh) {
    // Test 1x8 large 1D mesh shape
    // Shape: 0-1-2-3-4-5-6-7
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1}}, {1, {0, 2}}, {2, {1, 3}}, {3, {2, 4}}, {4, {3, 5}}, {5, {4, 6}}, {6, {5, 7}}, {7, {6}}};
    TestingMockControlPlane mock_cp(
        adjacency_data, tt::tt_metal::distributed::MeshShape(1, 8), {0, 1, 2, 3, 4, 5, 6, 7});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    // Verify all chip mappings for 1x8 mesh
    verify_physical_chip_ids(
        physical_chip_ids,
        8,
        {
            {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}  // horizontal line
        });
}

// Torus and ring support?
TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsLooped1DMesh) {
    GTEST_SKIP() << "Ring topology currently not supported";
    // Test 1x4 looped 1D mesh shape (torus-like)
    // Shape: 0-1-2-3-
    //        |       |
    //        --------
    // All chips have 2 neighbors (no corners in looped topology)
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 3}},  // Connected to both ends
        {1, {0, 2}},
        {2, {1, 3}},
        {3, {0, 2}}  // Connected to both ends
    };
    TestingMockControlPlane mock_cp(adjacency_data, tt::tt_metal::distributed::MeshShape(1, 4), {0, 1, 2, 3});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    verify_physical_chip_ids(physical_chip_ids, 4);
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsLooped2DMesh) {
    GTEST_SKIP() << "Torus topology currently not supported";
    // Test 2x2 looped 2D mesh shape (torus-like)
    // Shape: 0-1
    //        | |
    //        2-3
    // All chips have 4 neighbors (no corners in torus topology)
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 2, 1, 2}},  // Connected to all 4 directions (wrapped)
        {1, {0, 3, 0, 3}},  // Connected to all 4 directions (wrapped)
        {2, {3, 0, 3, 0}},  // Connected to all 4 directions (wrapped)
        {3, {2, 1, 2, 1}}   // Connected to all 4 directions (wrapped)
    };
    TestingMockControlPlane mock_cp(adjacency_data, tt::tt_metal::distributed::MeshShape(2, 2), {0, 1, 2, 3});

    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids();

    verify_physical_chip_ids(physical_chip_ids, 4);
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsWithStartingChipId) {
    // Test 3x3 mesh with custom northwest corner chip ID
    // Shape: 0-1-2
    //        | | |
    //        3-4-5
    //        | | |
    //        6-7-8
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 3}},
        {1, {0, 2, 4}},
        {2, {1, 5}},
        {3, {0, 4, 6}},
        {4, {1, 3, 5, 7}},
        {5, {2, 4, 8}},
        {6, {3, 7}},
        {7, {4, 6, 8}},
        {8, {5, 7}}};
    TestingMockControlPlane mock_cp(
        adjacency_data, tt::tt_metal::distributed::MeshShape(3, 3), {0, 1, 2, 3, 4, 5, 6, 7, 8});

    // Test with default northwest corner chip ID (should use chip 0)
    auto physical_chip_ids_default = mock_cp.get_mesh_physical_chip_ids();
    EXPECT_EQ(physical_chip_ids_default[0], 0) << "Default northwest corner chip should be 0";

    // Test with custom northwest corner chip ID (chip 4)
    auto physical_chip_ids_custom = mock_cp.get_mesh_physical_chip_ids(mock_cp.get_mesh_container(), 2);
    EXPECT_EQ(physical_chip_ids_custom[0], 2) << "Custom northwest corner chip should be 4";

    // Test with invalid northwest corner chip ID (should throw)
    EXPECT_THROW(
        { auto physical_chip_ids_invalid = mock_cp.get_mesh_physical_chip_ids(mock_cp.get_mesh_container(), 99); },
        std::exception)
        << "Should throw when northwest corner chip ID is not in mesh container";
}

}  // namespace tt::tt_fabric::fabric_router_tests
