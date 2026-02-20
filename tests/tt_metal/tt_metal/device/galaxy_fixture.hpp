// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {

class GalaxyFixture : public MeshDispatchFixture {
    bool SkipTestSuiteIfNotGalaxyMotherboard() {
        this->arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        return !(this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices >= 32);
    }

protected:
    void SetUp() override {
        this->DetectDispatchMode();
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            GTEST_SKIP() << "Not a galaxy mobo";
        }
        MeshDispatchFixture::SetUp();
    }

private:
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> device_ids_to_devices_;
};

class TGFixture : public MeshDispatchFixture {
    void SkipTestSuiteIfNotTG() {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "This test can only run on Wormhole B0";
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if ((num_devices != 32) or (num_pcie_devices != 4)) {
            GTEST_SKIP() << "This test can only run on TG";
        }
    }

protected:
    void SetUp() override {
        this->SkipTestSuiteIfNotTG();
        MeshDispatchFixture::SetUp();
    };

private:
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> device_ids_to_devices_;
};

}  // namespace tt::tt_metal
