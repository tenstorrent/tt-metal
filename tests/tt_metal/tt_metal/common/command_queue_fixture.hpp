// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <umd/device/types/arch.h>
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
#include "llrt.hpp"

namespace tt::tt_metal {

class CommandQueueFixture : public DispatchFixture<CommandQueueFixture> {
public:
    static bool WillSkip() {
        if (IsSlowDispatch()) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires Fast Dispatch"; }

    void SetUp() override {
        if (WillSkip()) {
            GTEST_SKIP() << GetSkipMessage();
        }
        DispatchFixture<CommandQueueFixture>::SetUp();
    }

    void TearDown() override { DispatchFixture<CommandQueueFixture>::TearDown(); }
};

class CommandQueueEventFixture : public CommandQueueFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoTearDownTestSuite();
    }

    void SetUp() override {
        // This test needs to start device counters at zero each time
        CommandQueueFixture::ResetTestSuite();
        CommandQueueFixture::SetUp();
    }
};

class CommandQueueBufferFixture : public CommandQueueFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoSetUpTestSuite(32 * 1024);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoTearDownTestSuite();
    }
};

class CommandQueueProgramFixture : public CommandQueueFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoSetUpTestSuite(32 * 1024);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoTearDownTestSuite();
    }
};

class CommandQueueTraceFixture : public CommandQueueFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoSetUpTestSuite(32 * 1024);
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoTearDownTestSuite();
    }
};

class CommandQueueSingleCardFixture : virtual public CommandQueueFixture {
private:
    inline static std::vector<tt::tt_metal::IDevice*> devices_under_test;

public:
    // Yes. Hiding parent devices_ as this fixture only exposes a single card
    std::vector<tt::tt_metal::IDevice*> devices_;

    // Push back the devices to be tested for Single Card fixture
    // To be used if a child class needs to call DoSetUpTestSuite() on the DispatchFixture themselves (e.g., with trace)
    static void SelectDevices() {
        const chip_id_t mmio_device_id = *tt::tt_metal::MetalContext::instance().get_cluster().mmio_chip_ids().begin();
        std::vector<chip_id_t> chip_ids;
        if (tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0) == BoardType::UBB) {
            for (chip_id_t id : tt::tt_metal::MetalContext::instance().get_cluster().user_exposed_chip_ids()) {
                chip_ids.push_back(id);
            }
        } else {
            chip_ids.push_back(mmio_device_id);
        }
        auto enable_remote_chip = getenv("TT_METAL_ENABLE_REMOTE_CHIP");
        devices_under_test.clear();
        if (enable_remote_chip) {
            for (const auto& [id, device] : GetDevicesMap()) {
                devices_under_test.push_back(device);
            }
        } else {
            for (const auto& chip_id : chip_ids) {
                devices_under_test.push_back(GetDevicesMap().at(chip_id));
            }
        }
    }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueFixture::DoSetUpTestSuite();
        SelectDevices();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        devices_under_test.clear();
        CommandQueueFixture::DoTearDownTestSuite();
    }

    void SetUp() override {
        CommandQueueFixture::SetUp();
        devices_ = devices_under_test;
    }

    void TearDown() override {
        CommandQueueFixture::TearDown();
        devices_.clear();
    }
};

class CommandQueueSingleCardBufferFixture : public CommandQueueSingleCardFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueSingleCardFixture::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        CommandQueueSingleCardFixture::DoTearDownTestSuite();
    }
};

class CommandQueueSingleCardTraceFixture : public CommandQueueSingleCardFixture {
public:
    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        log_info(tt::LogTest, "CommandQueueSingleCardTraceFixture: SetUpTestSuite");
        CommandQueueSingleCardFixture::DoSetUpTestSuiteWithTrace(90000000);
        SelectDevices();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        log_info(tt::LogTest, "CommandQueueSingleCardTraceFixture: TearDownTestSuite");
        CommandQueueSingleCardFixture::DoTearDownTestSuite();
    }
};

class CommandQueueSingleCardProgramFixture : public CommandQueueSingleCardFixture {};

class CommandQueueMultiDeviceFixture : public DispatchFixture<CommandQueueMultiDeviceFixture> {
public:
    static bool WillSkip() {
        if (IsSlowDispatch()) {
            return true;
        }
        auto num_devices = tt::tt_metal::GetNumAvailableDevices();
        if (num_devices < 2) {
            return true;
        }
        return false;
    }

    static std::string_view GetSkipMessage() { return "Requires >= 2 Devices and Fast Dispatch"; }

    static void SetUpTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<CommandQueueMultiDeviceFixture>::DoSetUpTestSuite();
    }

    static void TearDownTestSuite() {
        if (WillSkip()) {
            return;
        }
        DispatchFixture<CommandQueueMultiDeviceFixture>::DoTearDownTestSuite();
    }

    void SetUp() override { DispatchFixture<CommandQueueMultiDeviceFixture>::SetUp(); }
};

class CommandQueueMultiDeviceProgramFixture : public CommandQueueMultiDeviceFixture {};

class CommandQueueMultiDeviceBufferFixture : public CommandQueueMultiDeviceFixture {};

class CommandQueueOnFabricMultiDeviceFixture : public CommandQueueMultiDeviceFixture,
                                               public ::testing::WithParamInterface<tt::tt_metal::FabricConfig> {
public:
    void SetUp() override {
        if (tt::get_arch_from_string(tt::test_utils::get_umd_arch_name()) != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Dispatch on Fabric tests only applicable on Wormhole B0";
        }
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(true);
        // This will force dispatch init to inherit the FabricConfig param
        tt::tt_metal::detail::SetFabricConfig(GetParam(), FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE, 1);
        CommandQueueMultiDeviceFixture::SetUp();

        if (::testing::Test::IsSkipped()) {
            tt::tt_metal::detail::SetFabricConfig(FabricConfig::DISABLED);
        }
    }

    void TearDown() override {
        CommandQueueMultiDeviceFixture::TearDown();
        tt::tt_metal::detail::SetFabricConfig(FabricConfig::DISABLED);
        tt::tt_metal::MetalContext::instance().rtoptions().set_fd_fabric(false);
    }
};

}  // namespace tt::tt_metal
