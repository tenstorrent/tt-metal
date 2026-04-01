// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>

#include "gtest/gtest.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tests/tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/tt_metal/common/command_queue_fixture.hpp"

#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "hostdevcommon/common_values.hpp"
#include "common/tt_backend_api_types.hpp"

using namespace tt::tt_metal;  // For test

namespace ttnn {
class TTNNFixtureBase : public ::testing::Test {
protected:
    int trace_region_size_ = DEFAULT_TRACE_REGION_SIZE;
    int l1_small_size_ = DEFAULT_L1_SMALL_SIZE;
    tt::ARCH arch_ = tt::ARCH::Invalid;
    size_t num_devices_ = 0;

public:
    bool check_dispatch_mode() {
        auto* slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        return slow_dispatch == nullptr;
    }

    TTNNFixtureBase() : TTNNFixtureBase(DEFAULT_TRACE_REGION_SIZE, DEFAULT_L1_SMALL_SIZE) {}

    TTNNFixtureBase(int trace_region_size, int l1_small_size) :
        trace_region_size_(trace_region_size), l1_small_size_(l1_small_size), num_devices_(GetNumAvailableDevices()) {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    }
};

class TTNNUnitMeshCQSharedFixture : public ::tt::tt_metal::UnitMeshCQSingleCardSharedFixture {
protected:
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;

    void SetUp() override {
        ::tt::tt_metal::UnitMeshCQSingleCardSharedFixture::SetUp();
        device_ = devices_.empty() ? nullptr : devices_[0].get();
    }
};

class MultiCommandQueueSingleDeviceFixture : public TTNNFixtureBase {
protected:
    tt::tt_metal::distributed::MeshDevice* device_ = nullptr;
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> device_holder_;

    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Skipping test, since it can only be run in Fast Dispatch Mode.";
        }

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

// Suite-level shared T3K meshes (8× unit mesh, 2 CQs, Ethernet dispatch). Mirrors recovery behavior of
// UnitMeshCQSingleCardSharedFixture: one create per suite, recreate after failure, no per-test close.
class MultiCommandQueueT3KFixture : public TTNNFixtureBase {
protected:
    inline static std::map<tt::ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> shared_devs_;
    inline static bool devices_valid_ = false;
    inline static bool needs_recovery_ = false;

    std::map<tt::ChipId, std::shared_ptr<tt::tt_metal::distributed::MeshDevice>> devs;

    static void SetUpTestSuite() {
        if (suite_can_create_devices()) {
            try {
                create_shared_devs();
            } catch (const std::exception& e) {
                log_warning(tt::LogTest, "Failed to create shared T3K devices: {}", e.what());
            }
        }
    }

    static void TearDownTestSuite() { destroy_shared_devs(); }

    void SetUp() override {
        if (!check_dispatch_mode()) {
            GTEST_SKIP() << "Skipping test, since it can only be run in Fast Dispatch Mode.";
        }

        if (num_devices_ < 8 or arch_ != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi CQ test suite on non T3K machine.";
        }

        if (needs_recovery_ || !devices_valid_) {
            if (suite_can_create_devices()) {
                destroy_shared_devs();
                try {
                    create_shared_devs();
                } catch (const std::exception& e) {
                    log_warning(tt::LogTest, "Failed to recreate shared T3K devices: {}", e.what());
                }
            }
        }

        if (shared_devs_.empty()) {
            GTEST_SKIP() << "Skipping T3K Multi CQ test suite (shared devices not available).";
        }

        devs = shared_devs_;
    }

    void TearDown() override {
        if (HasFailure()) {
            needs_recovery_ = true;
        }
    }

private:
    static bool suite_can_create_devices() {
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
            return false;
        }
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            return false;
        }
        if (GetNumAvailableDevices() < 8) {
            return false;
        }
        return true;
    }

    static void create_shared_devs() {
        shared_devs_ = tt::tt_metal::distributed::MeshDevice::create_unit_meshes(
            {0, 1, 2, 3, 4, 5, 6, 7},
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            2,
            DispatchCoreConfig{DispatchCoreType::ETH});
        devices_valid_ = true;
        needs_recovery_ = false;
    }

    static void destroy_shared_devs() {
        for (auto& [id, device] : shared_devs_) {
            device->close();
        }
        shared_devs_.clear();
        devices_valid_ = false;
    }
};

}  // namespace ttnn
