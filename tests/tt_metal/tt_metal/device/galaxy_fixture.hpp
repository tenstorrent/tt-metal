// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device_pool.hpp>
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {

class GalaxyFixture : public MeshDispatchFixture {
protected:
    bool SkipTestSuiteIfNotGalaxyMotherboard() {
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        return !(this->arch_ == tt::ARCH::WORMHOLE_B0 && num_devices >= 32);
    }

    void SetUp() override {
        this->DetectDispatchMode();
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            GTEST_SKIP() << "Not a galaxy mobo";
        }
        MeshDispatchFixture::SetUp();
    }

private:
    std::map<chip_id_t, std::shared_ptr<distributed::MeshDevice>> device_ids_to_devices_;
};

class TGFixture : public GalaxyFixture {
protected:
    void SkipTestSuiteIfNotTG() {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            GTEST_SKIP() << "Not a galaxy mobo";
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 32 && num_pcie_devices == 4)) {
            GTEST_SKIP() << "This test can only run on TG";
        }
    }
};

}  // namespace tt::tt_metal
