// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <memory>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/multi_mesh_types.hpp>
#include <impl/context/metal_context.hpp>
#include "gmock/gmock.h"
#include <fmt/format.h>
#include "utils.hpp"

namespace tt::tt_fabric {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::tt::tt_metal::distributed::MeshShape;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::test::utils::ScopedEnvVar;

// RAII guard for managing mesh binding environment variables
class ScopedMeshBinding {
public:
    ScopedMeshBinding(const char* mesh_id, const char* host_rank)
        : mesh_id_guard_("TT_MESH_ID", mesh_id),
          host_rank_guard_("TT_HOST_RANK", host_rank) {}

    // Convenience constructor for numeric values
    ScopedMeshBinding(uint32_t mesh_id, uint32_t host_rank)
        : mesh_id_str_(std::to_string(mesh_id)),
          host_rank_str_(std::to_string(host_rank)),
          mesh_id_guard_("TT_MESH_ID", mesh_id_str_.c_str()),
          host_rank_guard_("TT_HOST_RANK", host_rank_str_.c_str()) {}

private:
    std::string mesh_id_str_;
    std::string host_rank_str_;
    ScopedEnvVar mesh_id_guard_;
    ScopedEnvVar host_rank_guard_;
};

// RAII helper that combines environment setup and control plane initialization
class ScopedControlPlane {
public:
    ScopedControlPlane(const std::string& mesh_desc,
                      MeshId mesh_id = MeshId{0},
                      int num_chips = 1,
                      std::optional<std::pair<uint32_t, uint32_t>> env_binding = std::nullopt)
        : mesh_binding_(env_binding ? std::make_unique<ScopedMeshBinding>(env_binding->first, env_binding->second) : nullptr),
          temp_file_(CreateTempFile(mesh_desc)) {

        auto chip_mapping = CreateChipMapping(mesh_id, num_chips);
        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(temp_file_, chip_mapping);
    }

    ~ScopedControlPlane() { std::remove(temp_file_.c_str()); }
    tt::tt_fabric::ControlPlane& get() { return tt::tt_metal::MetalContext::instance().get_control_plane(); }

private:
    static std::string CreateTempFile(const std::string& content) {
        static int counter = 0;
        std::string filename = "/tmp/test_mesh_desc_scoped_" + std::to_string(counter++) + ".yaml";
        std::ofstream file(filename);
        file << content;
        file.close();
        return filename;
    }

    static std::map<FabricNodeId, chip_id_t> CreateChipMapping(MeshId mesh_id, int num_chips) {
        std::map<FabricNodeId, chip_id_t> mapping;
        for (int i = 0; i < num_chips; ++i) {
            mapping[FabricNodeId(mesh_id, i)] = i;
        }
        return mapping;
    }

    std::unique_ptr<ScopedMeshBinding> mesh_binding_;
    std::string temp_file_;
};

// Mesh descriptor factory functions
namespace TestMeshDescriptors {

std::string CreateSingleChipMesh(const std::string& arch = "wormhole_b0", int mesh_id = 0) {
    return fmt::format(R"yaml(ChipSpec: {{
  arch: {},
  ethernet_ports: {{
    N: 2,
    E: 2,
    S: 2,
    W: 2,
  }}
}}

Board: [
  {{ name: SingleChip,
    type: Mesh,
    topology: [1, 1]}}
]

Mesh: [
{{
  id: {},
  board: SingleChip,
  device_topology: [1, 1],
  host_topology: [1, 1],
  host_ranks: [[0]]}}
]

Graph: []
)yaml", arch, mesh_id);
}

std::string CreateDualHostMesh(int rows = 2, int cols = 2, int mesh_id = 0) {
    return fmt::format(R"yaml(ChipSpec: {{
  arch: wormhole_b0,
  ethernet_ports: {{
    N: 2,
    E: 2,
    S: 2,
    W: 2,
  }}
}}

Board: [
  {{ name: DualHost,
    type: Mesh,
    topology: [1, {}]}}
]

Mesh: [
{{
  id: {},
  board: DualHost,
  device_topology: [{}, {}],
  host_topology: [2, 1],
  host_ranks: [[0], [1]]}}
]

Graph: []
)yaml", cols, mesh_id, rows, cols);
}

std::string CreateMultiMeshConfig(const std::vector<int>& mesh_ids) {
    std::string meshes;
    for (size_t i = 0; i < mesh_ids.size(); ++i) {
        if (i > 0) meshes += ",\n";
        meshes += fmt::format(R"({{
  id: {},
  board: SingleChip,
  device_topology: [1, 1],
  host_topology: [1, 1],
  host_ranks: [[0]]}})", mesh_ids[i]);
    }

    return fmt::format(R"yaml(ChipSpec: {{
  arch: wormhole_b0,
  ethernet_ports: {{
    N: 2,
    E: 2,
    S: 2,
    W: 2,
  }}
}}

Board: [
  {{ name: SingleChip,
    type: Mesh,
    topology: [1, 1]}}
]

Mesh: [
{}
]

Graph: []
)yaml", meshes);
}

// Legacy constants for backward compatibility
const std::string kSingleChipMeshDesc = CreateSingleChipMesh();
const std::string kDualHostMeshDesc = CreateDualHostMesh();
const std::string kMultipleMeshesDesc = CreateMultiMeshConfig({0, 1});

} // namespace TestMeshDescriptors

using namespace TestMeshDescriptors;

// Parameterized test data for MeshScope tests
struct MeshScopeTestParams {
    HostRankId host_rank;
    MeshShape expected_local_shape;
    MeshCoordinate expected_start;
    MeshCoordinate expected_end;
    std::string test_name;
};

// Test fixture for control plane API tests
class ControlPlaneLocalMeshBinding : public ::testing::Test {
protected:
    ~ControlPlaneLocalMeshBinding() override {
        // Clean up any temporary files
        for (const auto& file : temp_files_) {
            std::remove(file.c_str());
        }
    }

    std::string CreateTempMeshDescriptor(const std::string& content) {
        static int counter = 0;
        std::string filename = "/tmp/test_mesh_desc_" + std::to_string(counter++) + ".yaml";
        std::ofstream file(filename);
        file << content;
        file.close();
        temp_files_.push_back(filename);
        return filename;
    }

    // Helper to create standard chip mappings
    std::map<FabricNodeId, chip_id_t> CreateSequentialChipMapping(MeshId mesh_id, int num_chips) {
        std::map<FabricNodeId, chip_id_t> mapping;
        for (int i = 0; i < num_chips; ++i) {
            mapping[FabricNodeId(mesh_id, i)] = i;
        }
        return mapping;
    }

    // Helper function to set up control plane with a mesh descriptor
    tt::tt_fabric::ControlPlane& SetUpControlPlane(const std::string& mesh_desc, MeshId mesh_id = MeshId{0}, int num_chips = 1) {
        std::string temp_file = CreateTempMeshDescriptor(mesh_desc);
        auto chip_mapping = CreateSequentialChipMapping(mesh_id, num_chips);
        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(temp_file, chip_mapping);
        return tt::tt_metal::MetalContext::instance().get_control_plane();
    }

    // Helper to assert local binding matches expected values
    void AssertLocalBinding(const tt::tt_fabric::ControlPlane& control_plane, MeshId expected_mesh, HostRankId expected_host) {
        auto mesh_binding = control_plane.get_local_mesh_id_binding();
        auto host_binding = control_plane.get_local_host_rank_id_binding();

        ASSERT_TRUE(mesh_binding.has_value()) << "Local mesh binding not available";
        ASSERT_TRUE(host_binding.has_value()) << "Local host binding not available";
        EXPECT_EQ(*mesh_binding, expected_mesh);
        EXPECT_EQ(*host_binding, expected_host);
    }

    // Helper to assert no local binding is present
    void ExpectNoLocalBinding(const tt::tt_fabric::ControlPlane& control_plane) {
        EXPECT_FALSE(control_plane.get_local_mesh_id_binding().has_value());
        EXPECT_FALSE(control_plane.get_local_host_rank_id_binding().has_value());
    }

    std::vector<std::string> temp_files_;
};

TEST_F(ControlPlaneLocalMeshBinding, NoEnvironmentVariables) {
    auto& control_plane = SetUpControlPlane(kSingleChipMeshDesc);

    // Should return nullopt when environment variables are not set
    ExpectNoLocalBinding(control_plane);
}

TEST_F(ControlPlaneLocalMeshBinding, WithEnvironmentVariables) {
    ScopedMeshBinding env_guard(/*mesh_id*/0u, /*host_rank*/0u);
    auto& control_plane = SetUpControlPlane(kSingleChipMeshDesc);
    AssertLocalBinding(control_plane, MeshId{0}, HostRankId{0});
}

TEST_F(ControlPlaneLocalMeshBinding, PartialEnvironmentVariables) {
    std::string temp_file = CreateTempMeshDescriptor(kSingleChipMeshDesc);
    auto chip_mapping = CreateSequentialChipMapping(MeshId{0}, 1);

    // Test with only TT_MESH_ID set
    {
        ScopedEnvVar mesh_only("TT_MESH_ID", "0");

        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(temp_file, chip_mapping);
        auto& control_plane1 = tt::tt_metal::MetalContext::instance().get_control_plane();

        // Should return nullopt when only one variable is set
        ExpectNoLocalBinding(control_plane1);
    }

    // Test with only TT_HOST_RANK set
    {
        ScopedEnvVar host_only("TT_HOST_RANK", "0");

        // Reset control plane to pick up new environment variables
        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(temp_file, chip_mapping);
        auto& control_plane2 = tt::tt_metal::MetalContext::instance().get_control_plane();

        // Should return nullopt when only one variable is set
        ExpectNoLocalBinding(control_plane2);
    }
}

TEST_F(ControlPlaneLocalMeshBinding, GetPhysicalMeshShapeWithScopeDualHost) {
    // Test with host rank 0
    ScopedControlPlane scoped_cp(kDualHostMeshDesc, MeshId{0}, /*num_chips=*/4, /*env_binding=*/{{/*mesh_id=*/0, /*host_rank=*/0}});

    // Test global mesh shape
    auto global_shape = scoped_cp.get().get_physical_mesh_shape(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_shape, MeshShape(2, 2));

    // Verify local bindings
    AssertLocalBinding(scoped_cp.get(), MeshId{0}, HostRankId{0});

    // Test local mesh shape - should be 1x2 for host rank 0
    auto local_shape = scoped_cp.get().get_physical_mesh_shape(MeshId{0}, MeshScope::LOCAL);
    EXPECT_EQ(local_shape, MeshShape(1, 2));

    // Test default parameter (should be GLOBAL)
    auto default_shape = scoped_cp.get().get_physical_mesh_shape(MeshId{0});
    EXPECT_EQ(default_shape, global_shape);
}

TEST_F(ControlPlaneLocalMeshBinding, GetCoordRangeWithScopeDualHost) {
    // Test with host rank 1
    ScopedControlPlane scoped_cp(kDualHostMeshDesc, MeshId{0}, /*num_chips=*/4, /*env_binding=*/{{/*mesh_id=*/0, /*host_rank=*/1}});

    // Test global coordinate range
    auto global_range = scoped_cp.get().get_coord_range(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_range.start_coord(), MeshCoordinate(0, 0));
    EXPECT_EQ(global_range.end_coord(), MeshCoordinate(1, 1));
    EXPECT_EQ(global_range.shape(), MeshShape(2, 2));

    // Verify local bindings
    AssertLocalBinding(scoped_cp.get(), MeshId{0}, HostRankId{1});

    // Test local coordinate range - should be the bottom row for host rank 1
    auto local_range = scoped_cp.get().get_coord_range(MeshId{0}, MeshScope::LOCAL);
    EXPECT_EQ(local_range.start_coord(), MeshCoordinate(1, 0));
    EXPECT_EQ(local_range.end_coord(), MeshCoordinate(1, 1));
    EXPECT_EQ(local_range.shape(), MeshShape(1, 2));

    // Test default parameter (should be GLOBAL)
    auto default_range = scoped_cp.get().get_coord_range(MeshId{0});
    EXPECT_EQ(default_range, global_range);
}

TEST_F(ControlPlaneLocalMeshBinding, MultipleMeshes) {
    // Set environment variables for mesh 1 using RAII guard
    ScopedMeshBinding env_guard(/*mesh_id*/1, /*host_rank*/0);

    std::string temp_file = CreateTempMeshDescriptor(kMultipleMeshesDesc);

    // Create control plane with physical chip mapping for both meshes
    std::map<FabricNodeId, chip_id_t> chip_mapping;
    chip_mapping[FabricNodeId(MeshId{0}, 0)] = 0;
    chip_mapping[FabricNodeId(MeshId{1}, 0)] = 1;

    tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(temp_file, chip_mapping);
    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();

    // Should bind to mesh 1
    AssertLocalBinding(control_plane, MeshId{1}, HostRankId{0});

    // Test that we can query shapes for both meshes
    auto mesh0_shape = control_plane.get_physical_mesh_shape(MeshId{0});
    auto mesh1_shape = control_plane.get_physical_mesh_shape(MeshId{1});
    EXPECT_EQ(mesh0_shape, MeshShape(1, 1));
    EXPECT_EQ(mesh1_shape, MeshShape(1, 1));
}

TEST_F(ControlPlaneLocalMeshBinding, InvalidMeshId) {
    tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
      CreateTempMeshDescriptor(kSingleChipMeshDesc),
      CreateSequentialChipMapping(MeshId{0}, /*num_chips=*/1));

    auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    ExpectNoLocalBinding(control_plane);
    EXPECT_THROW(control_plane.get_physical_mesh_shape(MeshId{99}, MeshScope::GLOBAL), std::runtime_error);
}

TEST_F(ControlPlaneLocalMeshBinding, LocalMeshScopeQueryWithoutLocalBinding) {
    auto& control_plane = SetUpControlPlane(kSingleChipMeshDesc);
    ExpectNoLocalBinding(control_plane);
    EXPECT_NO_THROW(control_plane.get_coord_range(MeshId{0}, MeshScope::LOCAL));
}

// Parameterized test fixture for MeshScope tests
class MeshScopeParameterizedTest : public ControlPlaneLocalMeshBinding,
                                   public ::testing::WithParamInterface<MeshScopeTestParams> {};

// Parameterized test for get_physical_mesh_shape with different host ranks
TEST_P(MeshScopeParameterizedTest, GetPhysicalMeshShape) {
    const auto& params = GetParam();

    // Set up control plane with specified host rank
    ScopedControlPlane scoped_cp(kDualHostMeshDesc, MeshId{0}, /*num_chips=*/4, /*env_binding=*/std::pair<uint32_t, uint32_t>{/*mesh_id=*/0, /*host_rank=*/*params.host_rank});

    // Test global mesh shape (always 2x2)
    auto global_shape = scoped_cp.get().get_physical_mesh_shape(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_shape, MeshShape(2, 2));

    // Verify local bindings
    AssertLocalBinding(scoped_cp.get(), MeshId{0}, params.host_rank);

    // Test local mesh shape
    auto local_shape = scoped_cp.get().get_physical_mesh_shape(MeshId{0}, MeshScope::LOCAL);
    EXPECT_EQ(local_shape, params.expected_local_shape);
}

// Parameterized test for get_coord_range with different host ranks
TEST_P(MeshScopeParameterizedTest, GetCoordRange) {
    const auto& params = GetParam();

    // Set up control plane with specified host rank
    ScopedControlPlane scoped_cp(kDualHostMeshDesc, MeshId{0}, /*num_chips=*/4, /*env_binding=*/std::pair<uint32_t, uint32_t>{/*mesh_id=*/0, /*host_rank=*/*params.host_rank});

    // Test global coordinate range (always (0,0) to (1,1))
    auto global_range = scoped_cp.get().get_coord_range(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_range.start_coord(), MeshCoordinate(0, 0));
    EXPECT_EQ(global_range.end_coord(), MeshCoordinate(1, 1));
    EXPECT_EQ(global_range.shape(), MeshShape(2, 2));

    // Verify local bindings
    AssertLocalBinding(scoped_cp.get(), MeshId{0}, params.host_rank);

    // Test local coordinate range
    auto local_range = scoped_cp.get().get_coord_range(MeshId{0}, MeshScope::LOCAL);
    EXPECT_EQ(local_range.start_coord(), params.expected_start);
    EXPECT_EQ(local_range.end_coord(), params.expected_end);
    EXPECT_EQ(local_range.shape(), params.expected_local_shape);
}

// Test data for dual-host configuration (2x2 mesh with 2 hosts)
// Host rank 0 owns top row, host rank 1 owns bottom row
INSTANTIATE_TEST_SUITE_P(
    DualHostMeshScope,
    MeshScopeParameterizedTest,
    ::testing::Values(
        MeshScopeTestParams{
            HostRankId{0},
            MeshShape(1, 2),           // Host 0 owns 1x2 (top row)
            MeshCoordinate(0, 0),      // Start at (0,0)
            MeshCoordinate(0, 1),      // End at (0,1)
            "HostRank0"
        },
        MeshScopeTestParams{
            HostRankId{1},
            MeshShape(1, 2),           // Host 1 owns 1x2 (bottom row)
            MeshCoordinate(1, 0),      // Start at (1,0)
            MeshCoordinate(1, 1),      // End at (1,1)
            "HostRank1"
        }
    ),
    [](const testing::TestParamInfo<MeshScopeTestParams>& info) {
        return info.param.test_name;
    }
);

}  // namespace
}  // namespace tt::tt_fabric
