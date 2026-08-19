// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the finish command can wait for the last dprint.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {

namespace {

void RunTest(DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = mesh_device->get_devices()[0];

    // This tests prints only on a single core
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = {0, 0};

    // Run the program, use a large delay for the last print to emulate a long-running kernel.
    uint32_t clk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->id());
    uint32_t delay_cycles = clk_mhz * 4000000;  // 4 seconds
    const std::vector<uint32_t> args = {delay_cycles, xy_start.x, xy_start.y};
    fixture->RunProgram(mesh_device, "tests/tt_metal/tt_metal/test_kernels/device_print/print_with_wait.cpp", args);
    // Close system instantly after running to attempt to cut off prints.
    fixture->TearDownTestSuite();

    // Check the print log against expected output.
    vector<std::string> expected_output;
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            expected_output.push_back(fmt::format("({},{}) Before wait...", x, y));
            expected_output.push_back(fmt::format("({},{}) After wait...", x, y));
        }
    }
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, expected_output));
}

TEST_F(DevicePrintFixture, TensixTestPrintFinish) {
    auto mesh_devices = this->devices_;
    // Run only on the first device, as this tests disconnects devices and this can cause
    // issues on multi-device setups.
    this->RunTestOnDevice(RunTest, mesh_devices.at(0));
}

}  // namespace

}  // namespace CMAKE_UNIQUE_NAMESPACE
