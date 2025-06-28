// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "dispatch_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/device_pool.hpp>
#include <limits>
#include <algorithm>

namespace tt::tt_metal {

// Slow dispatch Device test suite
class DeviceFixture : public DispatchFixture<DeviceFixture> {
public:
    static bool WillSkip() {
        if (!IsSlowDispatch()) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires Slow Dispatch"; }

    std::pair<unsigned, unsigned> worker_grid_minimum_dims() {
        constexpr size_t UMAX = std::numeric_limits<unsigned>::max();
        std::pair<size_t, size_t> min_dims = {UMAX, UMAX};
        for (auto device : devices_) {
            auto coords = device->compute_with_storage_grid_size();
            min_dims.first = std::min(min_dims.first, coords.x);
            min_dims.second = std::min(min_dims.second, coords.y);
        }

        return min_dims;
    }

    void SetUp() override { DispatchFixture<DeviceFixture>::SetUp(); }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<DeviceFixture>::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<DeviceFixture>::DoTearDownTestSuite();
    }
};

class DeviceFixtureWithL1Small : public DeviceFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DeviceFixture::DoSetUpTestSuite(DEFAULT_TRACE_REGION_SIZE, 24 * 1024);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DeviceFixture::DoTearDownTestSuite();
    }
};

class DeviceSingleCardFixture : public DeviceFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DeviceFixture::DoSetUpTestSuiteWithNumberOfDevices(1);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DeviceFixture::DoTearDownTestSuite();
    }
};

class DeviceSingleCardBufferFixture : public DeviceFixture {};

class BlackholeSingleCardFixture : public DeviceFixture {
public:
    static bool WillSkip() {
        if (!IsSlowDispatch()) {
            return true;
        }

        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::BLACKHOLE) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires Slow Dispatch and Blackhole"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DeviceFixture::DoSetUpTestSuiteWithNumberOfDevices(1);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DeviceFixture::TearDownTestSuite();
    }

    void SetUp() override {
        if (WillSkip()) {
            GTEST_SKIP() << "This suite can only be run on Blackhole";
        }
        DeviceFixture::SetUp();
    }
};

class DeviceSingleCardFastSlowDispatchFixture : public DeviceSingleCardFixture {};

}  // namespace tt::tt_metal
