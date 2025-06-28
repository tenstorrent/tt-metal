// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.h>
#include "fabric_types.hpp"
#include "gtest/gtest.h"
#include "dispatch_fixture.hpp"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/device.hpp>
#include "umd/device/types/cluster_descriptor_types.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

class MultiCommandQueueSingleDeviceFixture : public DispatchFixture<MultiCommandQueueSingleDeviceFixture> {
public:
    static bool WillSkip() {
        if (IsSlowDispatch()) {
            return true;
        }

        auto num_cqs = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (num_cqs != 2) {
            return true;
        }

        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires fast dispatch and TT_METAL_GTEST_NUM_HW_CQS=2"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<MultiCommandQueueSingleDeviceFixture>::DoSetUpTestSuiteWithNumberOfDevices(1);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<MultiCommandQueueSingleDeviceFixture>::DoTearDownTestSuite();
    }
};

class MultiCommandQueueSingleDeviceEventFixture : public MultiCommandQueueSingleDeviceFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::TearDownTestSuite();
    }

    void SetUp() override {
        if (WillSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }
        // This test needs to start device counters at zero each time
        MultiCommandQueueSingleDeviceFixture::ResetTestSuite();
        MultiCommandQueueSingleDeviceFixture::SetUp();
    }
};

class MultiCommandQueueSingleDeviceBufferFixture : public MultiCommandQueueSingleDeviceFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::TearDownTestSuite();
    }
};

class MultiCommandQueueSingleDeviceProgramFixture : public MultiCommandQueueSingleDeviceFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::TearDownTestSuite();
    }
};

class MultiCommandQueueSingleDeviceTraceFixture : public MultiCommandQueueSingleDeviceFixture {
protected:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::DoSetUpTestSuite(32 * 1024, DEFAULT_L1_SMALL_SIZE, 1);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueSingleDeviceFixture::DoTearDownTestSuite();
    }
};

class MultiCommandQueueMultiDeviceFixture : public DispatchFixture<MultiCommandQueueMultiDeviceFixture> {
public:
    static bool WillSkip() {
        if (IsSlowDispatch()) {
            return true;
        }

        auto num_cqs = tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs();
        if (num_cqs != 2) {
            return true;
        }
        const size_t num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 2) {
            return true;
        }

        return false;
    }

    static std::string_view GetSkipMessage() {
        return "Requires fast dispatch, >= 2 devices, and TT_METAL_GTEST_NUM_HW_CQS=2";
    }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<MultiCommandQueueMultiDeviceFixture>::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<MultiCommandQueueMultiDeviceFixture>::DoTearDownTestSuite();
    }
};

class MultiCommandQueueMultiDeviceBufferFixture : public MultiCommandQueueMultiDeviceFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueMultiDeviceFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueMultiDeviceFixture::DoTearDownTestSuite();
    }
};

class MultiCommandQueueMultiDeviceEventFixture : public MultiCommandQueueMultiDeviceFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueMultiDeviceFixture::SetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueMultiDeviceFixture::DoTearDownTestSuite();
    }
};

// Multi Command Queue on Fabric Fixture
// This fixture will Open/Close devices on each test since different tests may have different FabricConfig
class MultiCommandQueueOnFabricMultiDeviceFixture : public MultiCommandQueueMultiDeviceFixture,
                                                    public ::testing::WithParamInterface<tt::tt_metal::FabricConfig> {
public:
    static bool WillSkip() {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            return true;
        }
        if (tt::tt_metal::MetalContext::instance().rtoptions().get_num_hw_cqs() != 2) {
            return true;
        }
        return false;
    }

    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp() override {
        if (WillSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }

        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(true);
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_metal::detail::SetFabricConfig(GetParam(), FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);

        MultiCommandQueueMultiDeviceFixture::SetUpTestSuite();
        MultiCommandQueueMultiDeviceFixture::SetUp();
    }

    void TearDown() override {
        if (WillSkip()) {
            return;
        }
        MultiCommandQueueMultiDeviceFixture::TearDown();
        MultiCommandQueueMultiDeviceFixture::TearDownTestSuite();

        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(false);
        tt::tt_metal::detail::SetFabricConfig(tt::tt_metal::FabricConfig::DISABLED);
    }
};

}  // namespace tt::tt_metal
