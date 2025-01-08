// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <math.h>
#include <algorithm>
#include <functional>
#include <random>

#include "gtest/gtest.h"

#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "tests/tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/host_api.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/distributed/mesh_device.hpp"

namespace ttnn {

class TTNNFixture : public ::testing::Test {
protected:
    tt::ARCH arch_;
    size_t num_devices_;

    void SetUp() override {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
    }

    void TearDown() override {}
};

class TTNNFixtureWithDevice : public TTNNFixture {
protected:
    tt::tt_metal::IDevice* device_ = nullptr;

    void SetUp() override {
        TTNNFixture::SetUp();
        device_ = tt::tt_metal::CreateDevice(0);
    }

    void TearDown() override {
        TTNNFixture::TearDown();
        tt::tt_metal::CloseDevice(device_);
    }

    tt::tt_metal::IDevice& getDevice() { return *device_; }
};

}  // namespace ttnn

namespace ttnn::distributed::test {

class T3kMultiDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }
        if (num_devices < 8 or arch != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
        }
        mesh_device_ = MeshDevice::create(MeshDeviceConfig(MeshShape{2, 4}, MeshType::Ring));
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }

        mesh_device_->close();
        mesh_device_.reset();
    }
    std::shared_ptr<MeshDevice> mesh_device_;
};

}  // namespace ttnn::distributed::test
