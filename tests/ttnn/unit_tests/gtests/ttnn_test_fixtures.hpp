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

/**
 * Fixture for tests requiring a single device, with suite-level device management.
 */
class TTNNFixtureWithDevice : public ::testing::Test {
private:
    static tt::tt_metal::IDevice* device_;  // Static device shared across tests
    static tt::ARCH arch_;                  // Architecture type
    static size_t num_devices_;             // Number of available devices

protected:
    /**
     * Suite-level setup: Initializes the device once for the entire test suite.
     */
    static void SetUpTestSuite() {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        device_ = tt::tt_metal::CreateDevice(
            /*device_id=*/0,
            /*num_hw_cqs=*/1,
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE);
    }

    /**
     * Suite-level cleanup: Closes the device after all tests in the suite complete.
     */
    static void TearDownTestSuite() {
        tt::tt_metal::CloseDevice(device_);
    }

    /**
     * Per-test setup: Sets the random seed for each test to ensure reproducibility.
     */
    void SetUp() override {
        std::srand(0);  // Reset random seed for consistent test results
    }

    /**
     * Default constructor: Provided for compatibility.
     */
    TTNNFixtureWithDevice() {}

    /**
     * Parameterized constructor: Retained for compatibility, though suite-level
     * setup uses default values for trace_region_size and l1_small_size.
     */
    TTNNFixtureWithDevice(int trace_region_size, int l1_small_size) {}
};

// Static member definitions
tt::tt_metal::IDevice* TTNNFixtureWithDevice::device_ = nullptr;
tt::ARCH TTNNFixtureWithDevice::arch_ = tt::ARCH::Invalid;
size_t TTNNFixtureWithDevice::num_devices_ = 0;

/**
 * Fixture for tests requiring a single device with multiple command queues.
 */
class MultiCommandQueueSingleDeviceFixture : public ::testing::Test {
protected:
    static tt::tt_metal::IDevice* device_;  // Static device shared across tests
    static tt::ARCH arch_;                  // Architecture type
    static size_t num_devices_;             // Number of available devices

    /**
     * Suite-level setup: Initializes a device with multi-command queue support.
     */
    static void SetUpTestSuite() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi CQ test suite, since it can only be run in Fast Dispatch Mode.";
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (arch_ == tt::ARCH::WORMHOLE_B0 && num_devices_ != 1) {
            tt::log_warning(tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
            dispatch_core_type = DispatchCoreType::ETH;
        }
        device_ = tt::tt_metal::CreateDevice(
            0, 2, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, DispatchCoreConfig{dispatch_core_type});
    }

    /**
     * Suite-level cleanup: Closes the device after all tests complete.
     */
    static void TearDownTestSuite() {
        tt::tt_metal::CloseDevice(device_);
    }
};

// Static member definitions
tt::tt_metal::IDevice* MultiCommandQueueSingleDeviceFixture::device_ = nullptr;
tt::ARCH MultiCommandQueueSingleDeviceFixture::arch_ = tt::ARCH::Invalid;
size_t MultiCommandQueueSingleDeviceFixture::num_devices_ = 0;

/**
 * Fixture for T3K-specific tests requiring multiple devices with multiple command queues.
 */
class MultiCommandQueueT3KFixture : public ::testing::Test {
protected:
    static std::map<chip_id_t, tt::tt_metal::IDevice*> devs;  // Map of devices for T3K tests
    static tt::ARCH arch_;                                    // Architecture type
    static size_t num_devices_;                               // Number of available devices

    /**
     * Suite-level setup: Initializes multiple devices for T3K tests.
     */
    static void SetUpTestSuite() {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi CQ test suite, since it can only be run in Fast Dispatch Mode.";
        }
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices_ < 8 || arch_ != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi CQ test suite on non T3K machine.";
        }
        // Enable Ethernet Dispatch for Multi-CQ tests
        devs = tt::tt_metal::detail::CreateDevices(
            {0, 1, 2, 3, 4, 5, 6, 7},
            2,
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            DispatchCoreConfig{DispatchCoreType::ETH});
    }

    /**
     * Suite-level cleanup: Closes all devices after the test suite completes.
     */
    static void TearDownTestSuite() {
        tt::tt_metal::detail::CloseDevices(devs);
    }
};

// Static member definitions
std::map<chip_id_t, tt::tt_metal::IDevice*> MultiCommandQueueT3KFixture::devs;
tt::ARCH MultiCommandQueueT3KFixture::arch_ = tt::ARCH::Invalid;
size_t MultiCommandQueueT3KFixture::num_devices_ = 0;

}  // namespace ttnn
