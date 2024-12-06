// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include "fabric_fixture.hpp"
#include "tt_fabric/control_plane.hpp"
#include "tt_fabric/mesh_graph.hpp"
#include "tt_fabric/routing_table_generator.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

TEST_F(ControlPlaneFixture, TestTGMeshGraphInit) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
        "tt_fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(tg_mesh_graph_desc_path.string());
}

TEST_F(ControlPlaneFixture, TestTGControlPlaneInit) {
  const std::filesystem::path tg_mesh_graph_desc_path = std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) / "tt_fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
  auto control_plane = std::make_unique<ControlPlane>(tg_mesh_graph_desc_path.string());
}

TEST_F(ControlPlaneFixture, TestT3kMeshGraphInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
        "tt_fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto mesh_graph_desc = std::make_unique<MeshGraph>(t3k_mesh_graph_desc_path.string());
}

TEST_F(ControlPlaneFixture, TestT3kControlPlaneInit) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
        "tt_fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(t3k_mesh_graph_desc_path.string());
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
