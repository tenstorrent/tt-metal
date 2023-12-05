// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "tt_metal/host_api.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/llrt/rtoptions.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the DPRINT server can detect an invalid core.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt::tt_metal;

TEST(DPrintErrorChecking, TestPrintInvalidCore) {
    auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    if (slow_dispatch) {
        TT_THROW("This suite can only be run with fast dispatch or TT_METAL_SLOW_DISPATCH_MODE unset");
        GTEST_SKIP();
    }

    // Skip for N300 for now (issue #3934).
    auto arch = tt::get_arch_from_string(tt::test_utils::get_env_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    auto num_pci_devices = tt::tt_metal::GetNumPCIeDevices();
    if (arch == tt::ARCH::WORMHOLE_B0 and
        num_devices == 2 and
        num_pci_devices == 1) {
        tt::log_info(tt::LogTest, "DPrint tests skipped on N300 for now.");
        GTEST_SKIP();
    }

    // Set DPRINT enabled on a mix of invalid and valid cores. Previously this would hang during
    // device setup, but not the print server should simply ignore the invalid cores.
    tt::llrt::OptionsG.set_dprint_cores({{0, 0}, {1, 1}, {100, 100}}); // Only (1,1) is valid.

    const int device_id = 0;
    Device* device = nullptr;
    device = tt::tt_metal::CreateDevice(device_id);

    // We expect that even though illegal worker cores were requested, device setup did not hang.
    // So just make sure that device setup worked and then close the device.
    EXPECT_TRUE(device != nullptr);
    tt::tt_metal::CloseDevice(device);
}
