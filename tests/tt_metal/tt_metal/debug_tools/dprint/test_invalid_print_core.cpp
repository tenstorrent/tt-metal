// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "debug_tools_fixture.hpp"
#include "gtest/gtest.h"
#include "impl/context/metal_context.hpp"
#include "umd/device/tt_core_coordinates.h"
#include "umd/device/types/xy_pair.h"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the DPRINT server can detect an invalid core.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt::tt_metal;

TEST_F(DPrintFixture, TensixTestPrintInvalidCore) {
    // Set DPRINT enabled on a mix of invalid and valid cores. Previously this would hang during
    // device setup, but not the print server should simply ignore the invalid cores.
    std::map<CoreType, std::vector<CoreCoord>> dprint_cores;
    dprint_cores[CoreType::WORKER] = {{0, 0}, {1, 1}, {100, 100}};
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_cores(
        tt::llrt::RunTimeDebugFeatureDprint, dprint_cores);

    // We expect that even though illegal worker cores were requested, device setup did not hang.
    // So just make sure that device setup worked and then close the device.
    for (IDevice* device : this->devices_) {
        EXPECT_TRUE(device != nullptr);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint, false);
}
