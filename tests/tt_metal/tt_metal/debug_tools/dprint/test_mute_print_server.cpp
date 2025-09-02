// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the DPRINT server can be muted/unmuted.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
const std::string golden_output =
R"(Printing int from arg: 0
Printing int from arg: 2)";

void RunTest(DPrintMeshFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

    // This tests prints only on a single core
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_one_int.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // A lambda to run the program w/ a given test number (used in the printing).
    auto run_program = [&](uint32_t test_number) {
        SetRuntimeArgs(program_, brisc_print_kernel_id, core, {test_number});
        fixture->RunProgram(mesh_device, workload);
    };

    // Run the program, prints should be enabled.
    run_program(0);

    // Disable the printing and run the program again.
    MetalContext::instance().dprint_server()->set_mute(true);
    run_program(1);

    // Re-enable prints and run the program one more time.
    MetalContext::instance().dprint_server()->set_mute(false);
    run_program(2);

    // Check the print log against golden output.
    EXPECT_TRUE(FilesMatchesString(DPrintMeshFixture::dprint_file_name, golden_output));
}
}
}

TEST_F(DPrintMeshFixture, TensixTestPrintMuting) {
    for (auto mesh_device : this->devices_) {
        this->RunTestOnDevice(CMAKE_UNIQUE_NAMESPACE::RunTest, mesh_device);
    }
}
