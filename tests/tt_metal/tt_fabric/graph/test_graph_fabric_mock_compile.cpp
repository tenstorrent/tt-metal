// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Mock fabric compile through the graph build session, for every base fabric config.
// Needs a mock cluster (TT_METAL_MOCK_CLUSTER_DESC_PATH, or tt-run with a mock rank binding).

#include <gtest/gtest.h>

#include <cstdlib>

#include <enchantum/enchantum.hpp>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "impl/context/metal_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric::graph {
namespace {

class GraphFabricMockCompile : public ::testing::TestWithParam<FabricConfig> {
protected:
    // TT_FABRIC_GRAPH_STOCK=1 runs the stock builder instead, for an A/B
    static void SetUpTestSuite() {
        tt::tt_metal::MetalContext::instance().rtoptions().set_fabric_graph_builder(
            std::getenv("TT_FABRIC_GRAPH_STOCK") == nullptr);
    }

    void SetUp() override {
        if (std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
            GTEST_SKIP() << "no mock cluster set";
        }
    }
    void TearDown() override { SetFabricConfig(FabricConfig::DISABLED); }
};

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

}  // namespace
}  // namespace tt::tt_fabric::graph
