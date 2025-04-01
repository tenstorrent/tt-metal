// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "rtoptions.hpp"
#include "tt_cluster.hpp"

namespace tt::tt_fabric {
namespace fabric_router_tests {

class ControlPlaneFixture : public ::testing::Test {
   protected:
       tt::ARCH arch_;
       void SetUp() override {
           auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
           if (not slow_dispatch) {
               tt::log_info(
                   tt::LogTest,
                   "Control plane test suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
               GTEST_SKIP();
           }
       }

       void TearDown() override {}
};

class BaseFabricFixture : public ::testing::Test {
protected:
    tt::ARCH arch_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> devices_map_;
    std::vector<tt::tt_metal::IDevice*> devices_;
    bool slow_dispatch_;

    void SetUpDevices(tt_metal::FabricConfig fabric_config) {
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch_) {
            tt::log_info(tt::LogTest, "Running fabric api tests with slow dispatch");
        } else {
            tt::log_info(tt::LogTest, "Running fabric api tests with fast dispatch");
        }
        // Set up all available devices
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        tt::tt_metal::detail::InitializeFabricConfig(fabric_config);
        devices_map_ = tt::tt_metal::detail::CreateDevices(ids);
        for (auto& [id, device] : devices_map_) {
            devices_.push_back(device);
        }
    }

    bool find_device_with_neighbor_in_direction(
        std::pair<mesh_id_t, chip_id_t>& src_mesh_chip_id,
        std::pair<mesh_id_t, chip_id_t>& dst_mesh_chip_id,
        RoutingDirection direction) {
        auto* control_plane = tt::Cluster::instance().get_control_plane();
        for (auto* device : devices_) {
            src_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());

            // Get neighbours within a mesh in the given direction
            auto neighbors =
                control_plane->get_intra_chip_neighbors(src_mesh_chip_id.first, src_mesh_chip_id.second, direction);
            if (neighbors.size() > 0) {
                dst_mesh_chip_id = {src_mesh_chip_id.first, neighbors[0]};
                return true;
            }
        }

        return false;
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
        tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
    }
};

class Fabric1DFixture : public BaseFabricFixture {
    void SetUp() override { this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_1D); }
};

class Fabric2DPullFixture : public BaseFabricFixture {
    void SetUp() override { this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D); }
};

class Fabric2DPushFixture : public BaseFabricFixture {
    void SetUp() override { this->SetUpDevices(tt::tt_metal::FabricConfig::FABRIC_2D_PUSH); }
};

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
