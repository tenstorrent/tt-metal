// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <gtest/gtest.h>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/mesh_graph_descriptor.hpp>
#include <tt-metalium/control_plane.hpp>
#include "impl/context/metal_context.hpp"

#include "tt_metal/fabric/physical_system_descriptor.hpp"
#include "tt_metal/fabric/topology_mapper.hpp"

namespace tt::tt_fabric {

class TopologyMapperTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple mesh graph for testing
        // This would typically be loaded from a descriptor file
        // For now, we'll create a minimal test setup
    }

    void TearDown() override {
        // Clean up any resources
    }
};

TEST_F(TopologyMapperTest, T3kMeshGraphTest) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";

    auto mesh_graph = MeshGraph(t3k_mesh_graph_desc_path.string());

    // Create PhysicalSystemDescriptor with proper parameters from MetalContext
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    constexpr bool using_mock_cluster_descriptor = false;
    constexpr bool run_discovery = true;

    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
        cluster.get_driver(), distributed_context, using_mock_cluster_descriptor, run_discovery);

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    local_mesh_binding.mesh_ids = {MeshId{0}};
    local_mesh_binding.host_rank = MeshHostRankId{0};

    // Test that TopologyMapper can be constructed with valid parameters
    // This is a basic smoke test
    auto topology_mapper = TopologyMapper(mesh_graph, physical_system_descriptor, local_mesh_binding);
}

TEST_F(TopologyMapperTest, T3kBigMeshTest) {
    const std::filesystem::path t3k_big_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_dual_host_mesh_graph_descriptor.yaml";

    auto mesh_graph = MeshGraph(t3k_big_mesh_graph_desc_path.string());

    // Create PhysicalSystemDescriptor with proper parameters from MetalContext
    auto distributed_context = tt::tt_metal::MetalContext::instance().get_distributed_context_ptr();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    constexpr bool using_mock_cluster_descriptor = false;
    constexpr bool run_discovery = true;

    auto physical_system_descriptor = tt::tt_metal::PhysicalSystemDescriptor(
        cluster.get_driver(), distributed_context, using_mock_cluster_descriptor, run_discovery);

    // Create a local mesh binding for testing
    LocalMeshBinding local_mesh_binding;
    if (*tt::tt_metal::MetalContext::instance().global_distributed_context().rank() == 0) {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{0};
    } else {
        local_mesh_binding.mesh_ids = {MeshId{0}};
        local_mesh_binding.host_rank = MeshHostRankId{1};
    }

    auto topology_mapper = TopologyMapper(mesh_graph, physical_system_descriptor, local_mesh_binding);
}

}  // namespace tt::tt_fabric
