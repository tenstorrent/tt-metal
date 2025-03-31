// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device_pool.hpp>
#include "multi_device_fixture.hpp"

namespace tt::tt_metal {

class GalaxyFixture : public MultiDeviceFixture {
protected:
    void SkipTestSuiteIfNotGalaxyMotherboard() {
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (!(this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices >= 32)) {
            GTEST_SKIP();
        }
    }

    void InitializeDevices() {
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        std::vector<chip_id_t> ids;
        for (uint32_t id = 0; id < num_devices; id++) {
            ids.push_back(id);
        }
        this->device_ids_to_devices_ = tt::tt_metal::detail::CreateDevices(ids);
        this->devices_ = tt::DevicePool::instance().get_all_active_devices();
    }

    void SetUp() override {
        MultiDeviceFixture::SetUp();
        this->DetectDispatchMode();
        this->SkipTestSuiteIfNotGalaxyMotherboard();
        this->InitializeDevices();
    }

    void TearDown() override {
        tt::tt_metal::detail::CloseDevices(this->device_ids_to_devices_);
        this->device_ids_to_devices_.clear();
        this->devices_.clear();
    }

private:
    std::map<chip_id_t, IDevice*> device_ids_to_devices_;
};

class TGFixture : public GalaxyFixture {
protected:
    void SkipTestSuiteIfNotTG() {
        this->SkipTestSuiteIfNotGalaxyMotherboard();
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 32 && num_pcie_devices == 4)) {
            GTEST_SKIP();
        }
    }

    void SetUp() override {
        MultiDeviceFixture::SetUp();
        this->DetectDispatchMode();
        this->SkipTestSuiteIfNotTG();
        this->InitializeDevices();
    }
};

class TGGFixture : public GalaxyFixture {
protected:
    void SkipTestSuiteIfNotTGG() {
        this->SkipTestSuiteIfNotGalaxyMotherboard();
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 64 && num_pcie_devices == 8)) {
            GTEST_SKIP();
        }
    }

    void SetUp() override {
        MultiDeviceFixture::SetUp();
        this->DetectDispatchMode();
        this->SkipTestSuiteIfNotTGG();
        this->InitializeDevices();
    }
};

}  // namespace tt::tt_metal
