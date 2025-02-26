// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include <tt-metalium/rtoptions.hpp>

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

}  // namespace fabric_router_tests

class FabricFixture : public ::testing::Test {
protected:
    tt::ARCH arch_;
    std::map<chip_id_t, IDevice*> devices_map_;
    std::vector<IDevice*> devices_;
    bool slow_dispatch_;

    void SetUp() override {
        slow_dispatch_ = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch_) {
            tt::log_info(
                tt::LogTest, "Fabric test suite can only be run with slow dispatch or TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        // Set up all available devices
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids;
        for (unsigned int id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        tt::tt_metal::detail::InitializeFabricSetting(tt::tt_metal::detail::FabricSetting::FABRIC);
        devices_map_ = tt::tt_metal::detail::CreateDevices(ids);
        for (auto& [id, device] : devices_map_) {
            devices_.push_back(device);
        }
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(devices_map_); }
};

}  // namespace tt::tt_fabric
