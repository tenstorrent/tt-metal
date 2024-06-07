#include <math.h>

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "gtest/gtest.h"

#include "ttnn/device.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"

namespace ttnn {

class TTNNFixture : public ::testing::Test {
   protected:
    tt::ARCH arch_;
    size_t num_devices_;

    void SetUp() override {
        std::srand(0);
        tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(true);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    }

    void TearDown() override { tt::Cluster::instance().set_internal_routing_info_for_ethernet_cores(false); }
};

class TTNNFixtureWithDevice : public TTNNFixture {
   protected:
    tt::tt_metal::Device* device_ = nullptr;

    void SetUp() override {
        TTNNFixture::SetUp();
        device_ = tt::tt_metal::CreateDevice(0);
    }

    void TearDown() override {
        TTNNFixture::TearDown();
        tt::tt_metal::CloseDevice(device_);
    }
};

}  // namespace ttnn
