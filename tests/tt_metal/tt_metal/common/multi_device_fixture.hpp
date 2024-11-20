// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

class MultiDeviceFixture : public ::testing::Test {
   protected:
    std::vector<tt::tt_metal::v1::DeviceHandle> devices_;
};

class N300DeviceFixture : public MultiDeviceFixture {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        const tt::ARCH arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() == 2 and
            tt::tt_metal::GetNumPCIeDevices() == 1) {
            std::vector<chip_id_t> ids;
            for (unsigned int id = 0; id < num_devices_; id++) {
                ids.push_back(id);
            }

            const auto &dispatch_core_type = tt::llrt::OptionsG.get_dispatch_core_type();
            tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_type);
            devices_ = tt::DevicePool::instance().get_all_active_devices();

        } else {
            GTEST_SKIP();
        }
    }

    void TearDown() override {
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false);
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    size_t num_devices_;
};
