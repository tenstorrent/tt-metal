// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <vector>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "intermesh_routing_test_utils.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {
namespace {

template <typename Fixture>
void validate_and_setup_control_plane_config(Fixture* fixture) {
    const char* mesh_id_str = std::getenv("TT_MESH_ID");
    const char* host_rank_str = std::getenv("TT_HOST_RANK");
    auto local_mesh_id = std::string(mesh_id_str);
    auto local_host_rank = std::string(host_rank_str);

    TT_FATAL(
        local_mesh_id.size() and local_host_rank.size(),
        "TT_MESH_ID and TT_HOST_RANK environment variables must be set for Multi-Host Fabric Tests.");

    auto chip_to_eth_coord_mapping = multihost_utils::get_physical_chip_mapping_from_eth_coords_mapping(
        fixture->get_eth_coord_mapping(), std::stoi(local_mesh_id));
    tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
        fixture->get_path_to_mesh_graph_desc(), chip_to_eth_coord_mapping);
    TT_FATAL(
        tt::tt_metal::MetalContext::instance().get_control_plane().system_has_intermesh_links(),
        "Multi-Host Routing tests require ethernet links to a remote host.");
    TT_FATAL(
        *(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) > 1,
        "Multi-Host Routing tests require multiple hosts in the system");
}

const std::vector<eth_coord_t>& get_eth_coords_for_t3k() {
    static const std::vector<eth_coord_t> t3k_eth_coords = {
        {0, 0, 0, 0, 0},
        {0, 1, 0, 0, 0},
        {0, 2, 0, 0, 0},
        {0, 3, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 0, 0},
        {0, 2, 1, 0, 0},
        {0, 3, 1, 0, 0}};

    return t3k_eth_coords;
}

}  // namespace

// Base fixture for Inter-Mesh Routing Fabric 2D tests.
class InterMeshRoutingFabric2DFixture : public BaseFabricFixture {
public:
    // This test fixture closes/opens devices on each test
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
    void SetUp() override {
        if (not system_supported()) {
            GTEST_SKIP() << "Skipping since this is not a supported system.";
        }

        validate_and_setup_control_plane_config(this);
        this->DoSetUpTestSuite(tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC);
    }

    void TearDown() override {
        if (system_supported()) {
            BaseFabricFixture::DoTearDownTestSuite();
        }
    }

    // Derived Classes (Fixtures specialized for topology/system) must define this
    virtual std::string get_path_to_mesh_graph_desc() = 0;
    virtual std::vector<std::vector<eth_coord_t>> get_eth_coord_mapping() = 0;
    // The derived fixture must infer if the current system is suitable for the requested
    // topology in the Mesh Graph, when implementing this function.
    virtual bool system_supported() = 0;
};

// Base fixture for Multi-Host MeshDevice tests relying on Inter-Mesh Routing.
class MultiMeshDeviceFabricFixture : public tt::tt_metal::T3000MeshDevice2DFabricFixture {
public:
    void SetUp() override {
        if (not system_supported()) {
            GTEST_SKIP() << "Skipping since this is not a supported system.";
        }

        validate_and_setup_control_plane_config(this);
        tt::tt_metal::T3000MeshDevice2DFabricFixture::SetUp();
    }

    void TearDown() override {
        if (system_supported()) {
            tt::tt_metal::T3000MeshDevice2DFabricFixture::TearDown();
        }
    }

    // Derived Classes (Fixtures specialized for topology/system) must define this
    virtual std::string get_path_to_mesh_graph_desc() = 0;
    virtual std::vector<std::vector<eth_coord_t>> get_eth_coord_mapping() = 0;
    virtual bool system_supported() = 0;
};

class InterMesh2x4FabricFixture : public InterMeshRoutingFabric2DFixture {
public:
    std::string get_path_to_mesh_graph_desc() override {
        return "tests/tt_metal/tt_fabric/custom_mesh_descriptors/t3k_2x2_mesh_graph_descriptor.yaml";
    }

    std::vector<std::vector<eth_coord_t>> get_eth_coord_mapping() override {
        return {
            {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}},
            {{0, 0, 0, 0, 0}, {0, 1, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}}};
    }

    bool system_supported() override {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        return cluster.user_exposed_chip_ids().size() == 4 &&
               *(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) == 2;
    }
};

template <typename Fixture>
class Dual2x4FabricFixture : public Fixture {
    std::string get_path_to_mesh_graph_desc() override {
        return "tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.yaml";
    }

    std::vector<std::vector<eth_coord_t>> get_eth_coord_mapping() override {
        return {get_eth_coords_for_t3k(), get_eth_coords_for_t3k()};
    }

    bool system_supported() override {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        return cluster.user_exposed_chip_ids().size() == 8 &&
               *(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) == 2;
    }
};

template <typename Fixture>
class NanoExaboxFabricFixture : public Fixture {
    std::string get_path_to_mesh_graph_desc() override {
        return "tests/tt_metal/tt_fabric/custom_mesh_descriptors/nano_exabox_mesh_graph_descriptor.yaml";
    }

    std::vector<std::vector<eth_coord_t>> get_eth_coord_mapping() override {
        return {
            get_eth_coords_for_t3k(),
            get_eth_coords_for_t3k(),
            get_eth_coords_for_t3k(),
            get_eth_coords_for_t3k(),
            get_eth_coords_for_t3k()};
    }

    bool system_supported() override {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        return cluster.user_exposed_chip_ids().size() == 8 &&
               *(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) == 5;
    }
};

using InterMeshDual2x4FabricFixture = Dual2x4FabricFixture<InterMeshRoutingFabric2DFixture>;
using MeshDeviceDual2x4Fixture = Dual2x4FabricFixture<MultiMeshDeviceFabricFixture>;

using IntermeshNanoExaboxFabricFixture = NanoExaboxFabricFixture<InterMeshRoutingFabric2DFixture>;
using MeshDeviceNanoExaboxFixture = NanoExaboxFabricFixture<MultiMeshDeviceFabricFixture>;

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
