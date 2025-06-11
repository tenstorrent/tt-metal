// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

class ControlPlaneFixture : public ::testing::Test {
   protected:
       tt::ARCH arch_;
       void SetUp() override {
           auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
           if (not slow_dispatch) {
               log_info(
                   tt::LogTest,
                   "Control plane test suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
               GTEST_SKIP();
           }
           // reserve max available planes
           uint8_t num_routing_planes = std::numeric_limits<uint8_t>::max();
           tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
               tt::tt_metal::FabricConfig::FABRIC_2D, num_routing_planes);
       }

       void TearDown() override {
           tt::tt_metal::MetalContext::instance().get_cluster().configure_ethernet_cores_for_fabric_routers(
               tt::tt_metal::FabricConfig::DISABLED);
       }
};

class BaseFabricFixture : public ::testing::Test {
public:
    tt::ARCH arch_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> devices_map_;
    std::vector<tt::tt_metal::IDevice*> devices_;
    bool slow_dispatch_;

    const std::vector<tt::tt_metal::IDevice*>& get_devices() const { return devices_; }
    void SetUpDevices(tt_metal::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes = std::nullopt) {
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch_) {
            log_info(tt::LogTest, "Running fabric api tests with slow dispatch");
        } else {
            log_info(tt::LogTest, "Running fabric api tests with fast dispatch");
        }
        // Set up all available devices
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        tt::tt_metal::detail::SetFabricConfig(fabric_config, num_routing_planes);
        devices_map_ = tt::tt_metal::detail::CreateDevices(ids);
        for (auto& [id, device] : devices_map_) {
            devices_.push_back(device);
        }
    }

    void RunProgramNonblocking(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            tt::tt_metal::detail::LaunchProgram(device, program, false);
        } else {
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::EnqueueProgram(cq, program, false);
        }
    }

    void WaitForSingleProgramDone(tt::tt_metal::IDevice* device, tt::tt_metal::Program& program) {
        if (this->slow_dispatch_) {
            // Wait for the program to finish
            tt::tt_metal::detail::WaitProgramDone(device, program);
        } else {
            // Wait for all programs on cq to finish
            tt::tt_metal::CommandQueue& cq = device->command_queue();
            tt::tt_metal::Finish(cq);
        }
    }

    void TearDown() override {
        tt::tt_metal::detail::CloseDevices(devices_map_);
        tt::tt_metal::detail::SetFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
    }
};

class Fabric1DFixture : public BaseFabricFixture {
    void SetUp() override { this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_1D); }
};

class Fabric2DFixture : public BaseFabricFixture {
    void SetUp() override { this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D); }
};

class Fabric2DDynamicFixture : public BaseFabricFixture {
    void SetUp() override { this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC); }
};

class CustomMeshGraphFabric2DDynamicFixture : public BaseFabricFixture {
public:
    void SetUp(
        const std::string& mesh_graph_desc_file,
        const std::map<FabricNodeId, chip_id_t>& logical_mesh_chip_id_to_physical_chip_id_mapping) {
        tt::tt_metal::MetalContext::instance().set_custom_control_plane_mesh_graph(
            mesh_graph_desc_file, logical_mesh_chip_id_to_physical_chip_id_mapping);
        this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D_DYNAMIC);
    }

private:
    void SetUp() override {}

    void TearDown() override {
        BaseFabricFixture::TearDown();
        tt::tt_metal::MetalContext::instance().set_default_control_plane_mesh_graph();
    }
};

class T3kCustomMeshGraphFabric2DDynamicFixture
    : public CustomMeshGraphFabric2DDynamicFixture,
      public testing::WithParamInterface<std::tuple<std::string, std::vector<std::vector<eth_coord_t>>>> {
    void SetUp() override {
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_cluster_type() != tt::ClusterType::T3K) {
            GTEST_SKIP();
        }
    }
};

struct McastRoutingInfo {
    RoutingDirection mcast_dir;
    uint32_t num_mcast_hops;
};

void RunTestUnicastRaw(
    BaseFabricFixture* fixture,
    uint32_t num_hops = 1,
    RoutingDirection direction = RoutingDirection::E,
    bool enable_fabric_tracing = false);

void RunTestUnicastConnAPI(
    BaseFabricFixture* fixture, uint32_t num_hops = 1, RoutingDirection direction = RoutingDirection::E);

void RunTestUnicastConnAPIRandom(BaseFabricFixture* fixture);

void RunTestMCastConnAPI(
    BaseFabricFixture* fixture,
    RoutingDirection fwd_dir = RoutingDirection::W,
    uint32_t fwd_hops = 1,
    RoutingDirection bwd_dir = RoutingDirection::E,
    uint32_t bwd_hops = 1);

void RunTestLineMcast(
    BaseFabricFixture* fixture, RoutingDirection unicast_dir, const std::vector<McastRoutingInfo>& mcast_routing_info);

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
