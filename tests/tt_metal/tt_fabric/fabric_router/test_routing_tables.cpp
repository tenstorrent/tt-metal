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
  auto path1 = control_plane->get_fabric_route(0, 0, 4, 31, 0);
  auto path2 = control_plane->get_fabric_route(0, 0, 4, 31, 1);
  auto path3 = control_plane->get_fabric_route(0, 0, 4, 31, 2);
  auto path4 = control_plane->get_fabric_route(0, 0, 4, 31, 3);
  auto path5 = control_plane->get_fabric_route(0, 0, 4, 31, 1);
}

TEST_F(ControlPlaneFixture, TestTGFabricRoutes) {
    const std::filesystem::path tg_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
        "tt_fabric/mesh_graph_descriptors/tg_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(tg_mesh_graph_desc_path.string());
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 3);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 4, 31, chan);
    }
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

TEST_F(ControlPlaneFixture, TestT3kFabricRoutes) {
    const std::filesystem::path t3k_mesh_graph_desc_path =
        std::filesystem::path(tt::llrt::OptionsG.get_root_dir()) /
        "tt_fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(t3k_mesh_graph_desc_path.string());
    auto valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 0);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 0, 7, chan);
    }
    valid_chans = control_plane->get_valid_eth_chans_on_routing_plane(0, 0, 1);
    for (auto chan : valid_chans) {
        auto path = control_plane->get_fabric_route(0, 0, 0, 7, chan);
    }
}

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
