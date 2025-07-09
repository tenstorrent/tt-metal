// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <memory>
#include <filesystem>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/multi_mesh_types.hpp>
#include <impl/context/metal_context.hpp>
#include "gmock/gmock.h"
#include <fmt/format.h>
#include "utils.hpp"
#include <umd/device/types/cluster_descriptor_types.h>

namespace tt::tt_fabric {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::IsEmpty;
using ::tt::tt_metal::distributed::MeshShape;
using ::tt::tt_metal::distributed::MeshCoordinate;
using ::tt::tt_metal::distributed::MeshCoordinateRange;
using ::tt::tt_metal::distributed::test::utils::ScopedEnvVar;
using ::tt::tt_metal::distributed::test::utils::TemporaryFile;

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

struct MeshScopeTestParams {
    HostRankId host_rank;
    MeshShape expected_local_shape;
    MeshCoordinate expected_start;
    MeshCoordinate expected_end;
    std::string test_name;
};

const std::string kDualHostMeshDesc =
    std::filesystem::path(::tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
    "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml";

// Define eth_coord mappings for the dual host mesh descriptor
// This is a 2x4 mesh split between 2 hosts, where:
// - Host 0 owns chips 0,1,4,5 (left half)
// - Host 1 owns chips 2,3,6,7 (right half)
const std::vector<std::vector<eth_coord_t>> kDualHostMeshEthCoords = {
    // Mesh 0 - all 8 chips arranged as 2x4
    {
        {0, 0, 0, 0, 0},  // chip 0 at (0,0)
        {0, 1, 0, 0, 0},  // chip 1 at (0,1)
        {0, 2, 0, 0, 0},  // chip 2 at (0,2)
        {0, 3, 0, 0, 0},  // chip 3 at (0,3)
        {0, 0, 1, 0, 0},  // chip 4 at (1,0)
        {0, 1, 1, 0, 0},  // chip 5 at (1,1)
        {0, 2, 1, 0, 0},  // chip 6 at (1,2)
        {0, 3, 1, 0, 0}   // chip 7 at (1,3)
    }};

// Helper function to get chip mapping from eth coords
std::map<FabricNodeId, chip_id_t> get_dual_host_chip_mapping() {
    const auto& cluster = ::tt::tt_metal::MetalContext::instance().get_cluster();
    std::map<FabricNodeId, chip_id_t> physical_chip_ids_mapping;
    for (std::uint32_t mesh_id = 0; mesh_id < kDualHostMeshEthCoords.size(); mesh_id++) {
        for (std::uint32_t chip_id = 0; chip_id < kDualHostMeshEthCoords[mesh_id].size(); chip_id++) {
            const auto& eth_coord = kDualHostMeshEthCoords[mesh_id][chip_id];
            physical_chip_ids_mapping.insert(
                {FabricNodeId(MeshId{mesh_id}, chip_id), cluster.get_physical_chip_id_from_eth_coord(eth_coord)});
        }
    }
    return physical_chip_ids_mapping;
}

// Test fixture for control plane API tests
class ControlPlaneLocalMeshBinding : public ::testing::Test {
protected:
    void SetUp() override {
        if (::tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != ::tt::ClusterType::T3K and
            ::tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != ::tt::ClusterType::N300_2x2) {
            GTEST_SKIP() << "Skipping test for non-T3K or N300_2x2 cluster";
        }
    }
};

TEST_F(ControlPlaneLocalMeshBinding, NoEnvironmentVariables) {
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), HostRankId{0});
}

TEST_F(ControlPlaneLocalMeshBinding, WithEnvironmentVariables) {
    ScopedMeshBinding env_guard(/*mesh_id*/0u, /*host_rank*/0u);
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), HostRankId{0});
}

// Test case to verify exception when both env vars are set but invalid
TEST_F(ControlPlaneLocalMeshBinding, InvalidEnvironmentVariablesMeshId) {
    // Set environment variables with invalid mesh ID
    ScopedMeshBinding env_guard(/*mesh_id*/ 99, /*host_rank*/ 0);

    EXPECT_THROW(
        {
            auto chip_mapping = get_dual_host_chip_mapping();
            auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
        },
        std::runtime_error);
}

TEST_F(ControlPlaneLocalMeshBinding, PartialEnvironmentVariables) {
    // Test with only TT_MESH_ID set
    {
        ScopedEnvVar mesh_only("TT_MESH_ID", "0");
        EXPECT_THROW(
            {
                auto chip_mapping = get_dual_host_chip_mapping();
                auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
            },
            std::runtime_error);
    }

    // Test with only TT_HOST_RANK set
    {
        ScopedEnvVar host_only("TT_HOST_RANK", "0");
        EXPECT_THROW(
            {
                auto chip_mapping = get_dual_host_chip_mapping();
                auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
            },
            std::runtime_error);
    }
}

TEST_F(ControlPlaneLocalMeshBinding, GetPhysicalMeshShapeWithScopeDualHost) {
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
    auto global_shape = control_plane->get_physical_mesh_shape(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_shape, MeshShape(2, 4));

    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), HostRankId{0});

    auto local_shape = control_plane->get_physical_mesh_shape(MeshId{0}, MeshScope::LOCAL);
    EXPECT_EQ(local_shape, MeshShape(2, 2));

    auto default_shape = control_plane->get_physical_mesh_shape(MeshId{0});
    EXPECT_EQ(default_shape, global_shape);
}

TEST_F(ControlPlaneLocalMeshBinding, GetCoordRangeWithScopeDualHost) {
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);

    auto global_range = control_plane->get_coord_range(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_range.start_coord(), MeshCoordinate(0, 0));
    EXPECT_EQ(global_range.end_coord(), MeshCoordinate(1, 3));
    EXPECT_EQ(global_range.shape(), MeshShape(2, 4));

    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), HostRankId{0});

    auto local_range = control_plane->get_coord_range(MeshId{0}, MeshScope::LOCAL);
    EXPECT_EQ(local_range.start_coord(), MeshCoordinate(0, 0));
    EXPECT_EQ(local_range.end_coord(), MeshCoordinate(1, 1));
    EXPECT_EQ(local_range.shape(), MeshShape(2, 2));

    auto default_range = control_plane->get_coord_range(MeshId{0});
    EXPECT_EQ(default_range, global_range);
}

TEST_F(ControlPlaneLocalMeshBinding, InvalidMeshId) {
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), HostRankId{0});
    EXPECT_THROW(control_plane->get_physical_mesh_shape(MeshId{99}, MeshScope::GLOBAL), std::runtime_error);
}

TEST_F(ControlPlaneLocalMeshBinding, LocalMeshScopeQueryWithoutExplicitBinding) {
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);
    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), HostRankId{0});
    EXPECT_NO_THROW(control_plane->get_coord_range(MeshId{0}, MeshScope::LOCAL));
}

// Parameterized test fixture for MeshScope tests
class MeshScopeParameterizedTest : public ControlPlaneLocalMeshBinding,
                                   public ::testing::WithParamInterface<MeshScopeTestParams> {};

// Parameterized test for get_physical_mesh_shape with different host ranks
TEST_P(MeshScopeParameterizedTest, GetPhysicalMeshShape) {
    const auto& params = GetParam();
    ScopedMeshBinding env_guard(/*mesh_id*/ 0u, /*host_rank*/ *params.host_rank);

    // Set up control plane with specified host rank
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);

    // Test global mesh shape (always 2x2)
    auto global_shape = control_plane->get_physical_mesh_shape(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_shape, MeshShape(2, 4));

    // Verify local bindings
    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), params.host_rank);

    // Test local mesh shape
    auto local_shape = control_plane->get_physical_mesh_shape(MeshId{0}, MeshScope::LOCAL);
    EXPECT_EQ(local_shape, params.expected_local_shape);
}

// Parameterized test for get_coord_range with different host ranks
TEST_P(MeshScopeParameterizedTest, GetCoordRange) {
    const auto& params = GetParam();
    ScopedMeshBinding env_guard(/*mesh_id*/ 0u, /*host_rank*/ *params.host_rank);

    // Set up control plane with specified host rank
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);

    // Test global coordinate range (always (0,0) to (1,1))
    auto global_range = control_plane->get_coord_range(MeshId{0}, MeshScope::GLOBAL);
    EXPECT_EQ(global_range.start_coord(), MeshCoordinate(0, 0));
    EXPECT_EQ(global_range.end_coord(), MeshCoordinate(1, 3));
    EXPECT_EQ(global_range.shape(), MeshShape(2, 4));

    // Verify local bindings
    EXPECT_EQ(control_plane->get_local_mesh_id_bindings()[0], MeshId{0});
    EXPECT_EQ(control_plane->get_local_host_rank_id_binding(), params.host_rank);

    // Test local coordinate range
    auto local_range = control_plane->get_coord_range(MeshId{0}, MeshScope::LOCAL);
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
            MeshShape(2, 2),       // Host 0 owns 2x2 (left half)
            MeshCoordinate(0, 0),  // Start at (0,0)
            MeshCoordinate(1, 1),  // End at (1,1)
            "HostRank0"},
        MeshScopeTestParams{
            HostRankId{1},
            MeshShape(2, 2),       // Host 1 owns 2x2 (right half)
            MeshCoordinate(0, 2),  // Start at (0,2)
            MeshCoordinate(1, 3),  // End at (1,3)
            "HostRank1"}),
    [](const testing::TestParamInfo<MeshScopeTestParams>& info) { return info.param.test_name; });

// Test case to verify LocalMeshBinding is inferred from MPI rank when env vars are not set
TEST_F(ControlPlaneLocalMeshBinding, InferLocalBindingFromMPIRank) {
    // Create a multi-mesh configuration with different host ranks
    std::string mesh_desc = R"yaml(ChipSpec: {
  arch: wormhole_b0,
  ethernet_ports: {
    N: 2,
    E: 2,
    S: 2,
    W: 2,
  }
}

Board: [
  { name: SingleChip,
    type: Mesh,
    topology: [1, 1]}
]

Mesh: [
{
  id: 0,
  board: SingleChip,
  device_topology: [1, 1],
  host_topology: [1, 1],
  host_ranks: [[0]]},
{
  id: 1,
  board: SingleChip,
  device_topology: [1, 1],
  host_topology: [1, 1],
  host_ranks: [[1]]}
]

Graph: []
)yaml";

    // Create control plane without environment variables
    auto chip_mapping = get_dual_host_chip_mapping();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(kDualHostMeshDesc, chip_mapping);

    // The control plane should have a valid binding inferred from MPI rank
    auto mesh_binding = control_plane->get_local_mesh_id_bindings()[0];
    auto host_binding = control_plane->get_local_host_rank_id_binding();

    // We can't predict which mesh/host will be assigned (depends on MPI rank),
    // but we can verify that a valid binding was established
    auto mesh_ids = control_plane->get_user_physical_mesh_ids();
    EXPECT_TRUE(std::find(mesh_ids.begin(), mesh_ids.end(), mesh_binding) != mesh_ids.end())
        << "Inferred mesh binding should be one of the configured meshes";

    // Verify that mesh operations work with the inferred binding
    EXPECT_NO_THROW(control_plane->get_physical_mesh_shape(mesh_binding, MeshScope::LOCAL));
    EXPECT_NO_THROW(control_plane->get_coord_range(mesh_binding, MeshScope::LOCAL));
}

}  // namespace
}  // namespace tt::tt_fabric
