// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <stdint.h>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
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

static void RunTest(DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

    // This tests prints only on a single core
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = {0, 0};

    KernelHandle brisc_print_kernel_id = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_with_wait.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Run the program, use a large delay for the last print to emulate a long-running kernel.
    uint32_t clk_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device->id());
    uint32_t delay_cycles = clk_mhz * 4000000;  // 4 seconds
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            const std::vector<uint32_t> args = { delay_cycles, x, y };
            SetRuntimeArgs(
                program_,
                brisc_print_kernel_id,
                CoreCoord{x, y},
                args
            );
        }
    }
    fixture->RunProgram(mesh_device, workload);
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
    EXPECT_TRUE(
        FileContainsAllStrings(
            DPrintMeshFixture::dprint_file_name,
            expected_output
        )
    );
}

TEST_F(DPrintMeshFixture, TensixTestPrintFinish) {
    auto mesh_devices = this->devices_;
    // Run only on the first device, as this tests disconnects devices and this can cause
    // issues on multi-device setups.
    this->RunTestOnDevice(RunTest, mesh_devices.at(0));
}
