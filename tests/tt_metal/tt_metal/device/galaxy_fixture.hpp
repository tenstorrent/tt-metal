// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device_pool.hpp>
#include "dispatch_fixture.hpp"

namespace tt::tt_metal {

class GalaxyFixture : public DispatchFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    bool SkipTestSuiteIfNotGalaxyMotherboard() {
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (!(tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) == tt::ARCH::WORMHOLE_B0 &&
              num_devices >= 32)) {
            log_info(tt::LogTest, "Not a Galaxy motherboard");
            return true;
        }
        return false;
    }

    void SetUp() override {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            GTEST_SKIP();
        }
        DispatchFixture::SetUp();
    }

    void TearDown() override {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            return;
        }
        DispatchFixture::TearDownTestSuite();
    }

private:
    std::map<chip_id_t, IDevice*> device_ids_to_devices_;
};

class TGFixture : public GalaxyFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    bool SkipTestSuiteIfNotTG() {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            return true;
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 32 && num_pcie_devices == 4)) {
            log_info(tt::LogTest, "Not a TG");
            return true;
        }
        return false;
    }

    void SetUp() override {
        if (this->SkipTestSuiteIfNotTG()) {
            GTEST_SKIP();
        }
        DispatchFixture::SetUp();
    }

    void TearDown() override { DispatchFixture::TearDownTestSuite(); }
};

// TGG is no longer supported
class DISABLED_TGGFixture : public GalaxyFixture {
protected:
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}

    bool SkipTestSuiteIfNotTGG() {
        if (this->SkipTestSuiteIfNotGalaxyMotherboard()) {
            return true;
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 64 && num_pcie_devices == 8)) {
            log_info(tt::LogTest, "Not a TGG");
            return true;
        }

        return false;
    }

    void SetUp() override {
        if (this->SkipTestSuiteIfNotTGG()) {
            GTEST_SKIP();
        }
        DispatchFixture::SetUp();
    }

    void TearDown() override { DispatchFixture::TearDownTestSuite(); }
};

}  // namespace tt::tt_metal
