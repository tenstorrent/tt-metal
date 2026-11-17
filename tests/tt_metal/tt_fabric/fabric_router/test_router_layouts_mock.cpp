// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Router channel layouts on a mock cluster (TT_METAL_MOCK_CLUSTER_DESC_PATH); skipped without one.

#include <gtest/gtest.h>

#include <cstdlib>
#include <array>
#include <optional>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/dispatch/dispatch_core_common.hpp"
#include "llrt/tt_cluster.hpp"
#include "tt_metal/fabric/erisc_datamover_builder.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"
#include "tt_metal/fabric/fabric_context.hpp"

namespace tt::tt_fabric {
namespace {

using tt::tt_metal::MetalContext;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
using tt::tt_metal::distributed::SystemMesh;

class RouterLayoutMockTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (std::getenv("TT_METAL_MOCK_CLUSTER_DESC_PATH") == nullptr) {
            GTEST_SKIP() << "needs a mock cluster (TT_METAL_MOCK_CLUSTER_DESC_PATH)";
        }
    }
    void TearDown() override {
        SetFabricConfig(FabricConfig::DISABLED);
        MetalContext::instance().rtoptions().set_enable_fabric_vc2(false);
    }
};

// A router with fewer senders than the fabric maxima (no Z sender, say) still has its VC2 sender's connection
// semaphore and producer cursor (runtime args) at its own VC2 index, after its own VC0 and VC1 senders.
TEST_F(RouterLayoutMockTest, Vc2BookkeepingAtRouterIndex) {
    MetalContext::instance().rtoptions().set_enable_fabric_vc2(true);
    SetFabricConfig(FabricConfig::FABRIC_2D);
    auto mesh = MeshDevice::create(MeshDeviceConfig(SystemMesh::instance().shape()));
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& fabric_context = control_plane.get_fabric_context();
    const auto& config = fabric_context.get_builder_context().get_fabric_router_config();
    if (config.num_used_sender_channels_per_vc[2] == 0) {
        mesh->close();
        GTEST_SKIP() << "VC2 not active on this cluster";
    }

    auto* device = mesh->get_devices().front();
    const auto node = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
    const auto [chan, direction] = *control_plane.get_active_fabric_eth_channels(node).begin();
    const auto eth_core = MetalContext::instance().get_cluster().get_soc_desc(device->id()).get_eth_core_for_channel(
        chan, CoordSystem::LOGICAL);
    std::array<size_t, builder_config::MAX_NUM_VCS> counts = {
        config.num_used_sender_channels_per_vc[0] - 1,
        config.num_used_sender_channels_per_vc[1],
        config.num_used_sender_channels_per_vc[2]};
    const size_t vc2 = counts[0] + counts[1];

    for (const bool worker_connection_mode : {false, true}) {
        tt::tt_metal::Program program;
        const auto builder = FabricEriscDatamoverBuilder::build(
            fabric_context,
            tt::tt_metal::get_core_type_from_config(MetalContext::instance().get_dispatch_core_config()),
            device,
            program,
            eth_core,
            node,
            control_plane.get_connected_mesh_chip_chan_ids(node, chan).first,
            config,
            std::vector<bool>(config.num_used_sender_channels, false),
            worker_connection_mode,
            direction,
            false,
            counts);
        const auto rt_args = builder.get_runtime_args();
        EXPECT_EQ(rt_args.at(vc2), config.sender_channels_connection_semaphore_address[vc2])
            << "worker_connection_mode=" << worker_connection_mode;
        EXPECT_EQ(
            rt_args.at(builder_config::num_max_sender_channels + vc2),
            config.sender_channels_buffer_index_semaphore_address[vc2])
            << "worker_connection_mode=" << worker_connection_mode;
    }
    mesh->close();
}

// A process whose fabric a fabric manager launched still has every local router's layout.
TEST_F(RouterLayoutMockTest, ManagedClientHasRouterLayouts) {
    SetFabricConfig(
        FabricConfig::FABRIC_2D,
        FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::nullopt,
        FabricTensixConfig::DISABLED,
        FabricUDMMode::DISABLED,
        FabricManagerMode::ENABLED);
    auto mesh = MeshDevice::create(MeshDeviceConfig(SystemMesh::instance().shape()));
    const auto& control_plane = MetalContext::instance().get_control_plane();
    const auto& builder_context = control_plane.get_fabric_context().get_builder_context();
    size_t routers = 0;
    for (const auto id : mesh->get_device_ids()) {
        const auto node = control_plane.get_fabric_node_id_from_physical_chip_id(id);
        for (const auto& [chan, direction] : control_plane.get_active_fabric_eth_channels(node)) {
            ++routers;
            EXPECT_NE(builder_context.find_router_channel_layout(node, chan), nullptr)
                << "router " << node << " channel " << static_cast<int>(chan);
        }
    }
    mesh->close();
    EXPECT_GT(routers, 0u);
}

}  // namespace
}  // namespace tt::tt_fabric
