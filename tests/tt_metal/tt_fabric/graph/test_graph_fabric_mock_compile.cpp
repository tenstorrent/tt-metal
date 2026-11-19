// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Mock fabric compile through the graph build session, for every base fabric config, and the stock fallback.
// Needs a mock cluster (TT_METAL_MOCK_CLUSTER_DESC_PATH, or tt-run with a mock rank binding).

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>
#include <string>

#include <enchantum/enchantum.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include <tt-metalium/experimental/fabric/inter_domain.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/graph/fabric_graph_inputs.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/graph/graph_fabric_build_session.hpp"

namespace tt::tt_fabric::graph {
namespace {

class GraphFabricMockTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
            GTEST_SKIP() << "needs a mock cluster: set TT_METAL_MOCK_CLUSTER_DESC_PATH, or run under tt-run with a "
                            "mock rank binding";
        }
        // TT_FABRIC_GRAPH_STOCK=1 runs the stock builder instead, for an A/B
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        was_graph_ = rtoptions.get_fabric_graph_builder();
        rtoptions.set_fabric_graph_builder(std::getenv("TT_FABRIC_GRAPH_STOCK") == nullptr);
    }
    void TearDown() override {
        if (IsSkipped()) {
            return;
        }
        SetFabricConfig(FabricConfig::DISABLED);
        tt::tt_metal::MetalContext::instance().rtoptions().set_fabric_graph_builder(was_graph_);
    }

    bool was_graph_ = false;
};

class GraphFabricMockCompile : public GraphFabricMockTest, public ::testing::WithParamInterface<FabricConfig> {};

TEST_P(GraphFabricMockCompile, BuildsRoutersOnEveryChip) {
    SetFabricConfig(GetParam());
    using namespace tt::tt_metal::distributed;
    auto mesh = MeshDevice::create(MeshDeviceConfig(SystemMesh::instance().shape()));

    const auto& builder_context =
        tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context().get_builder_context();
    for (const auto id : mesh->get_device_ids()) {
        EXPECT_GT(builder_context.get_num_fabric_initialized_routers(id), 0u) << "chip " << id;
    }
    mesh->close();
}

// Tensix mux and UDM fabrics aren't in the graph yet: the session falls back to the stock builder.
TEST_F(GraphFabricMockTest, TensixFabricsUseStockBuilder) {
    using namespace tt::tt_metal::distributed;
    const std::vector<std::pair<FabricTensixConfig, FabricUDMMode>> modes = {
        {FabricTensixConfig::DISABLED, FabricUDMMode::DISABLED},
        {FabricTensixConfig::MUX, FabricUDMMode::DISABLED},
        {FabricTensixConfig::UDM, FabricUDMMode::ENABLED}};
    for (const auto& [tensix, udm] : modes) {
        SetFabricConfig(
            FabricConfig::FABRIC_2D, FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, std::nullopt, tensix, udm);
        auto mesh = MeshDevice::create(MeshDeviceConfig(SystemMesh::instance().shape()));
        auto& fabric_context = tt::tt_metal::MetalContext::instance().get_control_plane().get_fabric_context();
        const bool graph = tensix == FabricTensixConfig::DISABLED && std::getenv("TT_FABRIC_GRAPH_STOCK") == nullptr;
        EXPECT_EQ(GraphFabricBuildSession(fabric_context).uses_graph(), graph) << enchantum::to_string(tensix);
        for (const auto id : mesh->get_device_ids()) {
            EXPECT_GT(fabric_context.get_builder_context().get_num_fabric_initialized_routers(id), 0u)
                << enchantum::to_string(tensix) << " chip " << id;
        }
        mesh->close();
        SetFabricConfig(FabricConfig::DISABLED);
    }
}

INSTANTIATE_TEST_SUITE_P(
    BaseFabricConfigs,
    GraphFabricMockCompile,
    ::testing::Values(
        FabricConfig::FABRIC_1D_NEIGHBOR_EXCHANGE,
        FabricConfig::FABRIC_1D,
        FabricConfig::FABRIC_1D_RING,
        FabricConfig::FABRIC_2D,
        FabricConfig::FABRIC_2D_TORUS_X,
        FabricConfig::FABRIC_2D_TORUS_Y,
        FabricConfig::FABRIC_2D_TORUS_XY),
    [](const auto& info) { return std::string(enchantum::to_string(info.param)); });

// Inter-domain on: builds every chip, then checks the perimeter inferred from the mock's unused links.
// TT_FABRIC_EXPECT_PERIMETER / _UNTRAINED=<n> pin the counts for the mock (and FSD) in use. For an untrained
// link, take an SC16 aisle D host's mock, delete one ethernet_connections_to_remote_devices entry, and run with
// that pod's FSD: the cable comes back as one untrained port.
class GraphFabricInterDomain : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
            GTEST_SKIP() << "no mock cluster set";
        }
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        rtoptions.set_fabric_graph_builder(true);
        rtoptions.set_fabric_inter_domain(true);
    }
    void TearDown() override {
        SetFabricConfig(FabricConfig::DISABLED);
        auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
        rtoptions.set_fabric_inter_domain(false);
        rtoptions.set_fabric_graph_builder(std::getenv("TT_FABRIC_GRAPH_STOCK") == nullptr);
    }
};

TEST_F(GraphFabricInterDomain, PerimeterLeavesTheDomain) {
    // relaxed: FSD and subset-MGD runs need it
    SetFabricConfig(FabricConfig::FABRIC_2D, FabricReliabilityMode::RELAXED_SYSTEM_HEALTH_SETUP_MODE);
    using namespace tt::tt_metal::distributed;
    auto mesh = MeshDevice::create(MeshDeviceConfig(SystemMesh::instance().shape()));

    const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
    const auto& mesh_graph = control_plane.get_mesh_graph();
    const auto& rtoptions = tt::tt_metal::MetalContext::instance().rtoptions();
    const auto links = unused_links(
        control_plane,
        tt::tt_metal::MetalContext::instance().get_cluster(),
        rtoptions.has_factory_system_descriptor_path() ? rtoptions.get_factory_system_descriptor_path() : "");
    const auto ports = perimeter_ports(links, ports_from_mesh_graph(mesh_graph, {}), mesh_graph, Topology::Mesh);

    size_t leaving = 0;
    for (const auto& l : links) {
        leaving += !l.peer_in_domain;
        for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(l.node)) {
            EXPECT_NE(chan, l.chan) << l.node << " chan " << int(l.chan) << " is a fabric channel";
        }
    }
    EXPECT_EQ(ports.size(), leaving);
    for (const auto& p : ports) {
        const auto it = std::find_if(
            links.begin(), links.end(), [&](const UnusedLink& l) { return l.node == p.node && l.chan == p.eth_chan; });
        ASSERT_NE(it, links.end());
        EXPECT_FALSE(it->peer_in_domain);
    }
    const auto untrained = std::count_if(ports.begin(), ports.end(), [](const auto& p) { return !p.link_trained; });
    std::cout << links.size() << " unused links, " << ports.size() << " perimeter ports, " << untrained << " untrained"
              << std::endl;
    if (const char* expect = std::getenv("TT_FABRIC_EXPECT_PERIMETER")) {
        EXPECT_EQ(ports.size(), std::stoul(expect));
    }
    if (const char* expect = std::getenv("TT_FABRIC_EXPECT_UNTRAINED")) {
        EXPECT_EQ(untrained, std::stol(expect));
    }

    // a router on every trained port, with a VC2 receiver for the peer to send into
    const auto api_ports = experimental::get_inter_domain_ports(*mesh);
    EXPECT_EQ(api_ports.size(), ports.size());
    for (const auto& port : api_ports) {
        if (!port.has_router) {
            continue;
        }
        const auto md = experimental::get_inter_domain_metadata(port, 1);
        EXPECT_NE(md.vc2_receiver_channel_addr, 0u) << port.node << " chan " << int(port.eth_chan);
        EXPECT_GT(md.vc2_receiver_channel_slots, 0u);
    }
    mesh->close();
}

}  // namespace
}  // namespace tt::tt_fabric::graph
