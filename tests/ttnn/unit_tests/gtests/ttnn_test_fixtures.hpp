// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <math.h>
#include <algorithm>
#include <functional>
#include <random>

#include "gtest/gtest.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tests/tt_metal/test_utils/env_vars.hpp"

#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "hostdevcommon/common_values.hpp"

using namespace tt::tt_metal;  // For test

namespace ttnn {
class TTNNFixtureBase : public ::testing::Test {
protected:
    int trace_region_size_ = DEFAULT_TRACE_REGION_SIZE;
    int l1_small_size_ = DEFAULT_L1_SMALL_SIZE;
    tt::ARCH arch_ = tt::ARCH::Invalid;
    size_t num_devices_ = 0;

public:
    void check_slow_dispatch() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping test, since it can only be run in Fast Dispatch Mode.";
        }
    }

public:
    TTNNFixtureBase() : TTNNFixtureBase(DEFAULT_TRACE_REGION_SIZE, DEFAULT_L1_SMALL_SIZE) { }

    TTNNFixtureBase(int trace_region_size, int l1_small_size) :
        trace_region_size_(trace_region_size), l1_small_size_(l1_small_size) {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = GetNumAvailableDevices();
    }
};

class TTNNFixtureWithDevice : public TTNNFixtureBase {
protected:
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;

    void SetUp() override {
        device_holder_ = ttnn::open_mesh_device(/*device_id=*/0, l1_small_size_, trace_region_size_);
        device_ = device_holder_.get();
    }

    void TearDown() override { device_->close(); }

    TTNNFixtureWithDevice() : TTNNFixtureBase() {}

    TTNNFixtureWithDevice(int trace_region_size, int l1_small_size) :
        TTNNFixtureBase(trace_region_size, l1_small_size) {}
};

class MultiCommandQueueSingleDeviceFixture : public TTNNFixtureBase {
protected:
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;

    void SetUp() override {
        check_slow_dispatch();

        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (arch_ == tt::ARCH::WORMHOLE_B0 and num_devices_ != 1) {
            log_warning(tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
            dispatch_core_type = DispatchCoreType::ETH;
        }
        device_holder_ = tt::tt_metal::distributed::MeshDevice::create_unit_mesh(
            0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 2, DispatchCoreConfig{dispatch_core_type});
        device_ = device_holder_.get();
    }

    void TearDown() override { device_->close(); }
};

class MultiCommandQueueT3KFixture : public TTNNFixtureBase {
protected:
    std::map<chip_id_t, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devs;

    void SetUp() override {
        check_slow_dispatch();

        if (num_devices_ < 8 or arch_ != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi CQ test suite on non T3K machine.";
        }

        // Enable Ethernet Dispatch for Multi-CQ tests.
        devs = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            {0, 1, 2, 3, 4, 5, 6, 7},
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            2,
            DispatchCoreConfig{DispatchCoreType::ETH});
    }

    void TearDown() override {
        for (auto& [_, dev] : devs) {
            dev->close();
        }
    }
};

}  // namespace ttnn
