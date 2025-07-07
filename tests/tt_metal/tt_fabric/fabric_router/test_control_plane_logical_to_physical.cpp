// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
// Now include the actual ControlPlane header with friend declaration
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t

#include <filesystem>
#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>

#include "fabric_fixture.hpp"

namespace tt::tt_fabric {

namespace fabric_router_tests {

// LogicalToPhysicalConversionFixture for adjacency map and lookup
class LogicalToPhysicalConversionFixture : public ::testing::Test {
protected:
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> test_adjacency_map;

    std::vector<chip_id_t> get_adjacent_chips(chip_id_t chip_id) const {
        auto it = test_adjacency_map.find(chip_id);
        if (it != test_adjacency_map.end()) {
            return it->second;
        }
        return {};
    }

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
};

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIdsWithConfigurableMock) {
    // Test 3x3 square mesh shape
    // Shape: 0-1-2
    //        | | |
    //        3-4-5
    //        | | |
    //        6-7-8
    test_adjacency_map = {
        {0, {1, 3}},
        {1, {0, 2, 4}},
        {2, {1, 5}},
        {3, {0, 4, 6}},
        {4, {1, 3, 5, 7}},
        {5, {2, 4, 8}},
        {6, {3, 7}},
        {7, {4, 6, 8}},
        {8, {5, 7}}};

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    tt::tt_metal::distributed::MeshShape mesh_shape(3, 3);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });

    auto physical_chip_ids = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);

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

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIds3x2RectangleMesh) {
    // Test 3x2 rectangle mesh shape
    // Shape: 0-1
    //        | |
    //        2-3
    //        | |
    //        4-5
    test_adjacency_map = {{0, {1, 2}}, {1, {0, 3}}, {2, {0, 3, 4}}, {3, {1, 2, 5}}, {4, {2, 5}}, {5, {3, 4}}};

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3, 4, 5};
    tt::tt_metal::distributed::MeshShape mesh_shape(3, 2);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });

    auto physical_chip_ids = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);

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

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIds4x8LargeMesh) {
    // Test 4x8 large rectangular mesh shape
    // Shape: 0-1-2-3-4-5-6-7
    //        | | | | | | | |
    //        8-9-A-B-C-D-E-F
    //        | | | | | | | |
    //        G-H-I-J-K-L-M-N
    //        | | | | | | | |
    //        O-P-Q-R-S-T-U-V
    test_adjacency_map = {
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

    std::set<chip_id_t> user_chip_ids = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    tt::tt_metal::distributed::MeshShape mesh_shape(4, 8);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });

    auto physical_chip_ids = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);

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

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIdsOverlappingConnections) {
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
    test_adjacency_map = {
        {0, {1, 3, 4}},        // Extra connection to 4
        {1, {0, 2, 3, 4, 5}},  // Extra connections to 3 and 5
        {2, {1, 5}},
        {3, {0, 1, 4, 6, 7}},  // Extra connections from 1 and to 7
        {4, {0, 1, 3, 5, 7}},  // Extra connections from 0 and to 7
        {5, {1, 2, 4, 7, 8}},  // Extra connections from 1 and to 7
        {6, {3, 7}},
        {7, {3, 4, 5, 6, 8}},  // Extra connections from 3, 4, 5
        {8, {5, 7}}};

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    tt::tt_metal::distributed::MeshShape mesh_shape(3, 3);

    // This test should fail because the topology has irregular connectivity
    // that doesn't match the expected 4-corner pattern for 2D meshes
    // The error message will now include ASCII visualization of the topology
    EXPECT_THROW(
        {
            auto topology_info = build_mesh_adjacency_map(
                user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });
            auto physical_chip_ids = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);
        },
        std::exception)
        << "Expected exception due to irregular topology with overlapping connections. "
        << "The error message should now include ASCII visualization of the mesh topology.";
}

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIdsMissingConnection) {
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
    test_adjacency_map = {
        {0, {1, 3}},
        {1, {0, 2, 4}},
        {2, {1, 5}},
        {3, {0, 4, 6}},
        {4, {1, 3, 5}},  // Missing connection to 7
        {5, {2, 4, 8}},
        {6, {3, 7}},
        {7, {6, 8}},  // Missing connection from 4
        {8, {5, 7}}};

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    tt::tt_metal::distributed::MeshShape mesh_shape(3, 3);

    // This test should fail because the missing connection creates an irregular
    // topology that doesn't match the expected 4-corner pattern for 2D meshes
    EXPECT_THROW(
        {
            auto topology_info = build_mesh_adjacency_map(
                user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });
            auto physical_chip_ids = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);
        },
        std::exception)
        << "Expected exception due to irregular topology with missing connection";
}

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIdsMissingChip) {
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
    test_adjacency_map = {
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

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3, 5, 6, 7, 8};  // 4 not in user_chip_ids
    tt::tt_metal::distributed::MeshShape mesh_shape(3, 3);

    // This test should fail because the missing chip creates an irregular
    // topology that doesn't match the expected 4-corner pattern for 2D meshes
    EXPECT_THROW(
        {
            auto topology_info = build_mesh_adjacency_map(
                user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });
            auto physical_chip_ids = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);
        },
        std::exception)
        << "Expected exception due to irregular topology with missing chip";
}

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIds4x1Mesh) {
    // Test 4x1 1D mesh shape
    // Shape: 0
    //        |
    //        1
    //        |
    //        2
    //        |
    //        3
    test_adjacency_map = {{0, {1}}, {1, {0, 2}}, {2, {1, 3}}, {3, {2}}};

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3};
    tt::tt_metal::distributed::MeshShape mesh_shape(4, 1);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });

    auto physical_chip_ids = convert_1d_mesh_adjacency_to_row_major_vector(topology_info);

    // Verify all chip mappings for 4x1 mesh
    verify_physical_chip_ids(
        physical_chip_ids,
        4,
        {
            {0, 0}, {1, 1}, {2, 2}, {3, 3}  // vertical line
        });
}

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIds1x8Mesh) {
    // Test 1x8 large 1D mesh shape
    // Shape: 0-1-2-3-4-5-6-7
    test_adjacency_map = {
        {0, {1}}, {1, {0, 2}}, {2, {1, 3}}, {3, {2, 4}}, {4, {3, 5}}, {5, {4, 6}}, {6, {5, 7}}, {7, {6}}};

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3, 4, 5, 6, 7};
    tt::tt_metal::distributed::MeshShape mesh_shape(1, 8);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });

    auto physical_chip_ids = convert_1d_mesh_adjacency_to_row_major_vector(topology_info);

    // Verify all chip mappings for 1x8 mesh
    verify_physical_chip_ids(
        physical_chip_ids,
        8,
        {
            {0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}  // horizontal line
        });
}

// Torus and ring support?
TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIdsLooped1DMesh) {
    GTEST_SKIP() << "Ring topology currently not supported";
    // Test 1x4 looped 1D mesh shape (torus-like)
    // Shape: 0-1-2-3-
    //        |       |
    //        --------
    // All chips have 2 neighbors (no corners in looped topology)
    test_adjacency_map = {
        {0, {1, 3}},  // Connected to both ends
        {1, {0, 2}},
        {2, {1, 3}},
        {3, {0, 2}}  // Connected to both ends
    };

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3};
    tt::tt_metal::distributed::MeshShape mesh_shape(1, 4);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });

    auto physical_chip_ids = convert_1d_mesh_adjacency_to_row_major_vector(topology_info);

    verify_physical_chip_ids(physical_chip_ids, 4);
}

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIdsLooped2DMesh) {
    GTEST_SKIP() << "Torus topology currently not supported";
    // Test 2x2 looped 2D mesh shape (torus-like)
    // Shape: 0-1
    //        | |
    //        2-3
    // All chips have 4 neighbors (no corners in torus topology)
    test_adjacency_map = {
        {0, {1, 2, 1, 2}},  // Connected to all 4 directions (wrapped)
        {1, {0, 3, 0, 3}},  // Connected to all 4 directions (wrapped)
        {2, {3, 0, 3, 0}},  // Connected to all 4 directions (wrapped)
        {3, {2, 1, 2, 1}}   // Connected to all 4 directions (wrapped)
    };

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3};
    tt::tt_metal::distributed::MeshShape mesh_shape(2, 2);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); });

    auto physical_chip_ids = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);

    verify_physical_chip_ids(physical_chip_ids, 4);
}

TEST_F(LogicalToPhysicalConversionFixture, TestGetMeshPhysicalChipIdsWithStartingChipId) {
    // Test 3x3 mesh with custom northwest corner chip ID
    // Shape: 0-1-2
    //        | | |
    //        3-4-5
    //        | | |
    //        6-7-8
    test_adjacency_map = {
        {0, {1, 3}},
        {1, {0, 2, 4}},
        {2, {1, 5}},
        {3, {0, 4, 6}},
        {4, {1, 3, 5, 7}},
        {5, {2, 4, 8}},
        {6, {3, 7}},
        {7, {4, 6, 8}},
        {8, {5, 7}}};

    std::set<chip_id_t> user_chip_ids = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    tt::tt_metal::distributed::MeshShape mesh_shape(3, 3);

    auto topology_info = build_mesh_adjacency_map(
        user_chip_ids, mesh_shape, [this](chip_id_t chip_id) { return this->get_adjacent_chips(chip_id); }, 4);

    // Test with default northwest corner chip ID (should use chip 0)
    auto physical_chip_ids_default = convert_2d_mesh_adjacency_to_row_major_vector(topology_info);
    EXPECT_EQ(physical_chip_ids_default[0], 0) << "Default northwest corner chip should be 0";

    // Test with custom northwest corner chip ID (chip 2)
    auto physical_chip_ids_custom = convert_2d_mesh_adjacency_to_row_major_vector(topology_info, 2);
    EXPECT_EQ(physical_chip_ids_custom[0], 2) << "Custom northwest corner chip should be 2";

    // Test with invalid northwest corner chip ID (should throw)
    EXPECT_THROW(
        { auto physical_chip_ids_invalid = convert_2d_mesh_adjacency_to_row_major_vector(topology_info, 99); },
        std::exception)
        << "Should throw when northwest corner chip ID is not in mesh container";
}

}  // namespace fabric_router_tests

}  // namespace tt::tt_fabric
