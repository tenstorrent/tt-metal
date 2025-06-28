
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <vector>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "intermesh_routing_test_utils.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

class InterMesh2x4Fabric2DFixture : public BaseFabricFixture {
public:
    void SetUp() override {
        if (not system_supported()) {
            GTEST_SKIP() << "Skipping since this is not a 2x4 system with intermesh links enabled";
        }
        local_binding_manager_.validate_local_mesh_id_and_host_rank();

        static const std::tuple<std::string, std::vector<std::vector<eth_coord_t>>> multi_mesh_2x4_chip_mappings =
            std::tuple{
                "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml",
                std::vector<std::vector<eth_coord_t>>{
                    {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
                    {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}}}};
        auto chip_to_eth_coord_mapping = multihost_utils::get_physical_chip_mapping_from_eth_coords_mapping(std::get<1>(multi_mesh_2x4_chip_mappings), 
            local_binding_manager_.get_local_mesh_id());
    
        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
            std::get<0>(multi_mesh_2x4_chip_mappings),
            chip_to_eth_coord_mapping);
        TT_FATAL(tt::tt_metal::MetalContext::instance().get_control_plane().system_has_intermesh_links(), "Multi-Host Routing tests require ethernet links to a remote host.");
        TT_FATAL(*(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) > 1, "Multi-Host Routing tests require multiple hosts in the system");
        this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC);
    }

    void TearDown() override {
        if (system_supported()) {
            BaseFabricFixture::TearDown();
            local_binding_manager_.clear_bindings();
            tt::tt_metal::MetalContext::instance().set_default_control_plane_mesh_graph();
            local_binding_manager_.set_bindings();
        }
    }

private:
    multihost_utils::LocalBindingManager local_binding_manager_;

    bool system_supported() {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        return cluster.user_exposed_chip_ids().size() == 4;
    }
};

class InterMeshDual2x4Fabric2DFixture : public BaseFabricFixture {
public:
    void SetUp() override {
        if (!system_supported()) {
            GTEST_SKIP() << "Skipping since this is not a dual T3K system with intermesh links enabled";
        }
        local_binding_manager_.validate_local_mesh_id_and_host_rank();

        static const std::tuple<std::string, std::vector<std::vector<eth_coord_t>>> multi_mesh_2x4_chip_mappings =
            std::tuple{
                "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.yaml",
                std::vector<std::vector<eth_coord_t>>{
                    {{0, 0, 0, 0, 0},
                    {0, 1, 0, 0, 0},
                    {0, 2, 0, 0, 0},
                    {0, 3, 0, 0, 0},
                    {0, 0, 1, 0, 0},
                    {0, 1, 1, 0, 0},
                    {0, 2, 1, 0, 0},
                    {0, 3, 1, 0, 0}},
                    
                    {{0, 0, 0, 0, 0},
                    {0, 1, 0, 0, 0},
                    {0, 2, 0, 0, 0},
                    {0, 3, 0, 0, 0},
                    {0, 0, 1, 0, 0},
                    {0, 1, 1, 0, 0},
                    {0, 2, 1, 0, 0},
                    {0, 3, 1, 0, 0}}}} ;
        auto chip_to_eth_coord_mapping = multihost_utils::get_physical_chip_mapping_from_eth_coords_mapping(std::get<1>(multi_mesh_2x4_chip_mappings), 
            local_binding_manager_.get_local_mesh_id());
        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
            std::get<0>(multi_mesh_2x4_chip_mappings),
            chip_to_eth_coord_mapping);
        TT_FATAL(tt::tt_metal::MetalContext::instance().get_control_plane().system_has_intermesh_links(), "Multi-Host Routing tests require ethernet links to a remote host.");
        TT_FATAL(*(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) > 1, "Multi-Host Routing tests require multiple hosts in the system");
        this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC);
    }

    void TearDown() override {
        if (system_supported()) {
            BaseFabricFixture::TearDown();
            local_binding_manager_.clear_bindings();
            tt::tt_metal::MetalContext::instance().set_default_control_plane_mesh_graph();
            local_binding_manager_.set_bindings();
        }
    }

private:
    multihost_utils::LocalBindingManager local_binding_manager_;
    bool system_supported() {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        return cluster.user_exposed_chip_ids().size() == 8;
    }
};

} // namespace fabric_router_tests
} // namespace tt::tt_fabric
