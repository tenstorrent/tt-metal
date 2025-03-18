// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "gtest/gtest.h"
#include "debug_tools_fixture.hpp"
#include <tt-metalium/host_api.hpp>
#include "rtoptions.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the DPRINT server can detect an invalid core.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt::tt_metal;

TEST_F(DPrintFixture, TensixTestPrintInvalidCore) {
    // Set DPRINT enabled on a mix of invalid and valid cores. Previously this would hang during
    // device setup, but not the print server should simply ignore the invalid cores.
    std::map<CoreType, std::vector<CoreCoord>> dprint_cores;
    dprint_cores[CoreType::WORKER] = {{0, 0}, {1, 1}, {100, 100}};
    tt::llrt::RunTimeOptions::get_instance().set_feature_cores(tt::llrt::RunTimeDebugFeatureDprint, dprint_cores);

    // We expect that even though illegal worker cores were requested, device setup did not hang.
    // So just make sure that device setup worked and then close the device.
    for (IDevice* device : this->devices_) {
        EXPECT_TRUE(device != nullptr);
    }
    tt::llrt::RunTimeOptions::get_instance().set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, false);
}
