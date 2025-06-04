// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <functional>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/utils.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the finish command can wait for the last dprint.
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal;

static void RunTest(DPrintFixture* fixture, IDevice* device) {
    // Set up program
    Program program = Program();

    // This tests prints only on a single core
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = {0, 0};
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_with_wait.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Run the program, use a large delay for the last print to emulate a long-running kernel.
    uint32_t clk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->id());
    uint32_t delay_cycles = clk_mhz * 2000000; // 2 seconds
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            const std::vector<uint32_t> args = { delay_cycles, x, y };
            SetRuntimeArgs(
                program,
                brisc_print_kernel_id,
                CoreCoord{x, y},
                args
            );
        }
    }
    fixture->RunProgram(device, program);
    // Close the device instantly after running to attempt to cut off prints.
    tt::tt_metal::CloseDevice(device);

    // Check the print log against expected output.
    vector<string> expected_output;
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            expected_output.push_back(fmt::format("({},{}) Before wait...", x, y));
            expected_output.push_back(fmt::format("({},{}) After wait...", x, y));
        }
    }
    EXPECT_TRUE(
        FileContainsAllStrings(
            DPrintFixture::dprint_file_name,
            expected_output
        )
    );
}

TEST_F(DPrintFixture, TensixTestPrintFinish) {
    auto devices = this->devices_;
    // Run only on the first device, as this tests disconnects devices and this can cause
    // issues on multi-device setups.
    this->RunTestOnDevice(RunTest, devices[0]);
}
