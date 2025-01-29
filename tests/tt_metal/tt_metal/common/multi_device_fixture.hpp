// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/host_api.hpp>
#include "llrt.hpp"
#include <tt-metalium/mesh_device.hpp>

#include "dispatch_fixture.hpp"
#include "umd/device/types/cluster_descriptor_types.h"
#include "tt_metal/test_utils/env_vars.hpp"

class MultiDeviceFixture : public DispatchFixture {
protected:
    void SetUp() override { this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()); }
};

class N300DeviceFixture : public MultiDeviceFixture {
protected:
    void SetUp() override {
        this->slow_dispatch_ = true;
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        if (!slow_dispatch) {
            tt::log_info(tt::LogTest, "This suite can only be run with TT_METAL_SLOW_DISPATCH_MODE set");
            this->slow_dispatch_ = false;
            GTEST_SKIP();
        }

        MultiDeviceFixture::SetUp();

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
        if (this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices == 2 && num_pci_devices == 1) {
            std::vector<chip_id_t> ids;
            for (chip_id_t id = 0; id < num_devices; id++) {
                ids.push_back(id);
            }

            const auto& dispatch_core_config = tt::llrt::RunTimeOptions::get_instance().get_dispatch_core_config();
            tt::DevicePool::initialize(ids, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, dispatch_core_config);
            this->devices_ = tt::DevicePool::instance().get_all_active_devices();
        } else {
            GTEST_SKIP();
        }
    }
};

class T3000MultiDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        using tt::tt_metal::distributed::MeshDevice;
        using tt::tt_metal::distributed::MeshDeviceConfig;
        using tt::tt_metal::distributed::MeshShape;

        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi-Device test suite, since it can only be run in Fast Dispatch Mode.";
        }
        if (num_devices < 8 or arch != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi-Device test suite on non T3K machine.";
        }
        mesh_device_ = MeshDevice::create(MeshDeviceConfig{.mesh_shape = MeshShape{2, 4}});
    }

    void TearDown() override {
        if (!mesh_device_) {
            return;
        }

        mesh_device_->close();
        mesh_device_.reset();
    }
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device_;
};
