// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device_pool.hpp>
#include "dispatch_fixture.hpp"

namespace tt::tt_metal {

class GalaxyFixture : public DispatchFixture<GalaxyFixture> {
public:
    static bool WillSkip() {
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        // Galaxy mobo has at least 32 devices and Wormhole B0
        if (!(arch == tt::ARCH::WORMHOLE_B0 && num_devices >= 32)) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires a galaxy mobo"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture::DoTearDownTestSuite();
    }

    void SetUp() override {
        if (WillSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }
        DispatchFixture::SetUp();
    }
};

class TGFixture : public GalaxyFixture {
public:
    static bool WillSkip() {
        if (GalaxyFixture::WillSkip()) {
            return true;
        }

        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 32 && num_pcie_devices == 4)) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires a TG"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture::DoTearDownTestSuite();
    }

    void SetUp() override {
        if (WillSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }
        DispatchFixture::SetUp();
    }
};

class TGGFixture : public GalaxyFixture {
protected:
    static bool WillSkip() {
        if (GalaxyFixture::WillSkip()) {
            return true;
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        const size_t num_pcie_devices = tt::tt_metal::GetNumPCIeDevices();
        if (!(num_devices == 64 && num_pcie_devices == 8)) {
            return true;
        }

        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires a TGG"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture::DoTearDownTestSuite();
    }

    void SetUp() override {
        if (WillSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }
        DispatchFixture::SetUp();
    }
};

}  // namespace tt::tt_metal
