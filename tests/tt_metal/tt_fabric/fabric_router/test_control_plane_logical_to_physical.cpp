// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>

// This is a hack to allow unit tests to access private members of the ControlPlane class
#define private public
#include <tt-metalium/control_plane.hpp>
#undef private

#include <filesystem>
#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>

#include "fabric_fixture.hpp"

namespace tt::tt_fabric::fabric_router_tests {

// Unified Configurable Mock ControlPlane class for testing get_mesh_physical_chip_ids
class ConfigurableMockControlPlane : public ControlPlane {
public:
    explicit ConfigurableMockControlPlane(const std::string& mesh_graph_desc_file) :
        ControlPlane(mesh_graph_desc_file) {}

    explicit ConfigurableMockControlPlane(
        const std::string& mesh_graph_desc_file,
        const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) :
        ControlPlane(mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping) {}

    void configure_test_data(
        const std::unordered_map<chip_id_t, std::vector<chip_id_t>>& adjacency_map,
        const std::set<chip_id_t>& user_chip_ids = {}) {
        adjacency_data_ = adjacency_map;
        user_exposed_chip_ids_ = user_chip_ids;
    }

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
        return {0, 1, 2, 3, 4, 5, 6, 7};
    }

private:
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data_;
    std::set<chip_id_t> user_exposed_chip_ids_;
};

// Fixture for ControlPlane logical to physical mapping tests
using ControlPlane = ::tt::tt_fabric::ControlPlane;

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsWithConfigurableMock) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data);
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(3, 3);
    EXPECT_EQ(physical_chip_ids.size(), 9);
    EXPECT_EQ(physical_chip_ids[0], 0);
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";

    // Verify corner mappings (NW, NE, SW, SE corners)
    EXPECT_EQ(physical_chip_ids[0], 0);
    EXPECT_EQ(physical_chip_ids[2], 2);
    EXPECT_EQ(physical_chip_ids[6], 6);
    EXPECT_EQ(physical_chip_ids[8], 8);

    // Verify edge mappings
    EXPECT_EQ(physical_chip_ids[1], 1);
    EXPECT_EQ(physical_chip_ids[3], 3);
    EXPECT_EQ(physical_chip_ids[5], 5);
    EXPECT_EQ(physical_chip_ids[7], 7);

    // Verify center mapping
    EXPECT_EQ(physical_chip_ids[4], 4);
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds1x1Mesh) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

    // Test 1x1 mesh shape (single chip)
    // Shape: 0
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {{0, {}}};
    mock_cp.configure_test_data(adjacency_data, {0});
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(1, 1);
    EXPECT_EQ(physical_chip_ids.size(), 1);
    EXPECT_EQ(physical_chip_ids[0], 0);
}
TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds3x2RectangleMesh) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

    // Test 3x2 rectangle mesh shape
    // Shape: 0-1
    //        | |
    //        2-3
    //        | |
    //        4-5
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1, 2}}, {1, {0, 3}}, {2, {0, 3, 4}}, {3, {1, 2, 5}}, {4, {2, 5}}, {5, {3, 4}}};
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3, 4, 5});
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(3, 2);
    EXPECT_EQ(physical_chip_ids.size(), 6);
    EXPECT_EQ(physical_chip_ids[0], 0);
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";

    // Verify corner mappings (NW, NE, SW, SE corners)
    EXPECT_EQ(physical_chip_ids[0], 0);
    EXPECT_EQ(physical_chip_ids[1], 1);
    EXPECT_EQ(physical_chip_ids[4], 4);
    EXPECT_EQ(physical_chip_ids[5], 5);

    // Verify edge mappings
    EXPECT_EQ(physical_chip_ids[2], 2);
    EXPECT_EQ(physical_chip_ids[3], 3);
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds4x8LargeMesh) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31});
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(4, 8);
    EXPECT_EQ(physical_chip_ids.size(), 32);
    EXPECT_EQ(physical_chip_ids[0], 0);
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";

    // Verify corner mappings (NW, NE, SW, SE corners)
    EXPECT_EQ(physical_chip_ids[0], 0);
    EXPECT_EQ(physical_chip_ids[7], 7);
    EXPECT_EQ(physical_chip_ids[24], 24);
    EXPECT_EQ(physical_chip_ids[31], 31);

    // Verify some edge mappings
    EXPECT_EQ(physical_chip_ids[1], 1);
    EXPECT_EQ(physical_chip_ids[8], 8);
    EXPECT_EQ(physical_chip_ids[15], 15);
    EXPECT_EQ(physical_chip_ids[25], 25);

    // Verify some interior mappings
    EXPECT_EQ(physical_chip_ids[9], 9);
    EXPECT_EQ(physical_chip_ids[17], 17);
    EXPECT_EQ(physical_chip_ids[26], 26);
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsOverlappingConnections) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3, 4, 5, 6, 7, 8});

    // This test should fail because the topology has irregular connectivity
    // that doesn't match the expected 4-corner pattern for 2D meshes
    EXPECT_THROW(
        { auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(3, 3); }, std::exception)
        << "Expected exception due to irregular topology with overlapping connections";
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsMissingConnection) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3, 4, 5, 6, 7, 8});

    // This test should fail because the missing connection creates an irregular
    // topology that doesn't match the expected 4-corner pattern for 2D meshes
    EXPECT_THROW(
        { auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(3, 3); }, std::exception)
        << "Expected exception due to irregular topology with missing connection";
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsMissingChip) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3, 5, 6, 7, 8});  // 4 not in user_chip_ids

    // This test should fail because the missing chip creates an irregular
    // topology that doesn't match the expected 4-corner pattern for 2D meshes
    EXPECT_THROW(
        { auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(3, 3); }, std::exception)
        << "Expected exception due to irregular topology with missing chip";
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds4x1Mesh) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3});
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(4, 1);
    EXPECT_EQ(physical_chip_ids.size(), 4);
    EXPECT_EQ(physical_chip_ids[0], 0);
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";

    // Verify endpoint mappings (for 1D mesh, these are the "corners")
    EXPECT_EQ(physical_chip_ids[0], 0);
    EXPECT_EQ(physical_chip_ids[3], 3);

    // Verify interior mappings
    EXPECT_EQ(physical_chip_ids[1], 1);
    EXPECT_EQ(physical_chip_ids[2], 2);
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIds1x8Mesh) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

    // Test 1x8 large 1D mesh shape
    // Shape: 0-1-2-3-4-5-6-7
    std::unordered_map<chip_id_t, std::vector<chip_id_t>> adjacency_data = {
        {0, {1}}, {1, {0, 2}}, {2, {1, 3}}, {3, {2, 4}}, {4, {3, 5}}, {5, {4, 6}}, {6, {5, 7}}, {7, {6}}};
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3, 4, 5, 6, 7});
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(1, 8);
    EXPECT_EQ(physical_chip_ids.size(), 8);
    EXPECT_EQ(physical_chip_ids[0], 0);
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";

    // Verify endpoint mappings (for 1D mesh, these are the "corners")
    EXPECT_EQ(physical_chip_ids[0], 0);
    EXPECT_EQ(physical_chip_ids[7], 7);

    // Verify some interior mappings
    EXPECT_EQ(physical_chip_ids[1], 1);
    EXPECT_EQ(physical_chip_ids[3], 3);
    EXPECT_EQ(physical_chip_ids[5], 5);
    EXPECT_EQ(physical_chip_ids[6], 6);
}

// Torus and ring support?
TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsLooped1DMesh) {
    GTEST_SKIP() << "Torus topology currently not supported";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3});
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(1, 4);
    EXPECT_EQ(physical_chip_ids.size(), 4);
    EXPECT_EQ(physical_chip_ids[0], 0);
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";
}

TEST_F(ControlPlaneFixture, TestGetMeshPhysicalChipIdsLooped2DMesh) {
    GTEST_SKIP() << "Torus topology currently not supported";
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    ConfigurableMockControlPlane mock_cp(mesh_graph_desc_path.string());

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
    mock_cp.configure_test_data(adjacency_data, {0, 1, 2, 3});
    auto physical_chip_ids = mock_cp.get_mesh_physical_chip_ids(2, 2);
    EXPECT_EQ(physical_chip_ids.size(), 4);
    EXPECT_EQ(physical_chip_ids[0], 0);
    for (size_t i = 0; i < physical_chip_ids.size(); ++i) {
        EXPECT_NE(physical_chip_ids[i], static_cast<chip_id_t>(-1)) << "Chip at index " << i << " is not mapped";
    }
    std::unordered_set<chip_id_t> unique_chips(physical_chip_ids.begin(), physical_chip_ids.end());
    EXPECT_EQ(unique_chips.size(), physical_chip_ids.size()) << "Duplicate chip IDs found in mapping";
}

}  // namespace tt::tt_fabric::fabric_router_tests
