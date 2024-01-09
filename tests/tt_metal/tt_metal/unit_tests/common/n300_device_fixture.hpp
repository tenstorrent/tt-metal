// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/test_utils/env_vars.hpp"

class N300DeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (not slow_dispatch) {
            TT_THROW("This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            GTEST_SKIP();
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() == 2 and
            tt::tt_metal::GetNumPCIeDevices() == 1) {
            for (unsigned int id = 0; id < num_devices_; id++) {
                auto* device = tt::tt_metal::CreateDevice(id);
                devices_.push_back(device);
            }
            tt::Cluster::instance().launch_internal_routing_for_ethernet_cores();

        } else {
            GTEST_SKIP();
        }
    }

    void TearDown() override {
        for (unsigned int id = 0; id < devices_.size(); id++) {
            tt::tt_metal::CloseDevice(devices_.at(id));
        }
    }

    std::vector<tt::tt_metal::Device*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;
};
