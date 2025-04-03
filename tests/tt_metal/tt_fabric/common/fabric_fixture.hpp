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
#include "impl/context/metal_context.hpp"

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
public:
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
        chip_id_t& src_physical_device_id,
        chip_id_t& dst_physical_device_id,
        RoutingDirection direction) {
        auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
        for (auto* device : devices_) {
            src_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());

            // Get neighbours within a mesh in the given direction
            auto neighbors =
                control_plane->get_intra_chip_neighbors(src_mesh_chip_id.first, src_mesh_chip_id.second, direction);
            if (neighbors.size() > 0) {
                src_physical_device_id = device->id();
                dst_mesh_chip_id = {src_mesh_chip_id.first, neighbors[0]};
                dst_physical_device_id = control_plane->get_physical_chip_id_from_mesh_chip_id(dst_mesh_chip_id);
                return true;
            }
        }

        return false;
    }

    // Find a device with enough neighbours in the specified direction
    bool find_device_with_neighbor_in_multi_direction(
        std::pair<mesh_id_t, chip_id_t>& src_mesh_chip_id,
        std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>& dst_mesh_chip_ids_by_dir,
        chip_id_t& src_physical_device_id,
        std::unordered_map<RoutingDirection, std::vector<chip_id_t>>& dst_physical_device_ids_by_dir,
        const std::unordered_map<RoutingDirection, uint32_t>& mcast_hops) {
        auto control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();

        // Find a device with enough neighbours in the specified direction
        bool connection_found = false;
        for (auto* device : devices_) {
            src_mesh_chip_id = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());
            std::unordered_map<RoutingDirection, std::vector<std::pair<mesh_id_t, chip_id_t>>>
                temp_end_mesh_chip_ids_by_dir;
            std::unordered_map<RoutingDirection, std::vector<chip_id_t>> temp_physical_end_device_ids_by_dir;
            connection_found = true;
            for (auto [routing_direction, num_hops] : mcast_hops) {
                bool direction_found = true;
                auto& temp_end_mesh_chip_ids = temp_end_mesh_chip_ids_by_dir[routing_direction];
                auto& temp_physical_end_device_ids = temp_physical_end_device_ids_by_dir[routing_direction];
                uint32_t curr_mesh_id = src_mesh_chip_id.first;
                uint32_t curr_chip_id = src_mesh_chip_id.second;
                for (uint32_t i = 0; i < num_hops; i++) {
                    auto neighbors =
                        control_plane->get_intra_chip_neighbors(curr_mesh_id, curr_chip_id, routing_direction);
                    if (neighbors.size() > 0) {
                        temp_end_mesh_chip_ids.emplace_back(curr_mesh_id, neighbors[0]);
                        temp_physical_end_device_ids.push_back(
                            control_plane->get_physical_chip_id_from_mesh_chip_id(temp_end_mesh_chip_ids.back()));
                        curr_mesh_id = temp_end_mesh_chip_ids.back().first;
                        curr_chip_id = temp_end_mesh_chip_ids.back().second;
                    } else {
                        direction_found = false;
                        break;
                    }
                }
                if (!direction_found) {
                    connection_found = false;
                    break;
                }
            }
            if (connection_found) {
                src_physical_device_id = device->id();
                dst_mesh_chip_ids_by_dir = std::move(temp_end_mesh_chip_ids_by_dir);
                dst_physical_device_ids_by_dir = std::move(temp_physical_end_device_ids_by_dir);
                break;
            }
        }
        return connection_found;
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
