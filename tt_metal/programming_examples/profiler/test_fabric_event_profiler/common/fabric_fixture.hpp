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

class BaseFabricFixture {
public:
    tt::ARCH arch_;
    std::map<chip_id_t, tt::tt_metal::IDevice*> devices_map_;
    std::vector<tt::tt_metal::IDevice*> devices_;
    bool slow_dispatch_;

    const std::vector<tt::tt_metal::IDevice*>& get_devices() const { return devices_; }
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

    void TearDown() {
        tt::tt_metal::detail::CloseDevices(devices_map_);
        tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
    }
};

// tests fabric tracing on a small number of unicast transfers on 1D fabric
int RunFabricProfilerUnicastTest(
    BaseFabricFixture* fixture, uint32_t num_hops = 1, RoutingDirection direction = RoutingDirection::E);
}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
