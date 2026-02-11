// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <typeinfo>
#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "fabric_fixture.hpp"
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/distributed_context.hpp>
#include "tt_metal/fabric/fabric_host_utils.hpp"
#include "tests/tt_metal/tt_fabric/common/utils.hpp"

namespace {

constexpr auto k_FabricConfig = tt::tt_fabric::FabricConfig::FABRIC_2D;
constexpr auto k_ReliabilityMode = tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE;

std::unique_ptr<tt::tt_fabric::ControlPlane> make_control_plane(const std::filesystem::path& graph_desc) {
    auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& distributed_context = tt::tt_metal::MetalContext::instance().full_world_distributed_context();
    auto control_plane = std::make_unique<tt::tt_fabric::ControlPlane>(
        cluster, rtoptions, hal, distributed_context, graph_desc.string(), k_FabricConfig, k_ReliabilityMode);
    control_plane->configure_routing_tables_for_fabric_ethernet_channels();
    return control_plane;
}

}  // namespace

namespace tt::tt_fabric::fabric_router_tests {

TEST_F(ControlPlaneFixture, TestCustom2x2ControlPlaneInit) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/n300_2x2_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(mesh_graph_desc_path);

    check_asic_mapping_against_golden("TestCustom2x2ControlPlaneInit", "ControlPlaneFixture_Custom2x2");
}

TEST_F(ControlPlaneFixture, TestCustom2x2MeshAPIs) {
    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    auto user_meshes = control_plane.get_user_physical_mesh_ids();
    EXPECT_EQ(user_meshes.size(), 1);
    EXPECT_EQ(user_meshes[0], MeshId{0});
    EXPECT_EQ(
        control_plane.get_physical_mesh_shape(MeshId{0}),
        tt::tt_metal::distributed::MeshShape(2, 2));
}

TEST_F(ControlPlaneFixture, TestCustom2x2ControlPlaneInitMGD2) {
    const std::filesystem::path mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/n300_2x2_mesh_graph_descriptor.textproto";
    [[maybe_unused]] auto control_plane = make_control_plane(mesh_graph_desc_path);

    check_asic_mapping_against_golden("TestCustom2x2ControlPlaneInitMGD2", "ControlPlaneFixture_Custom2x2");
}

}  // namespace tt::tt_fabric::fabric_router_tests
