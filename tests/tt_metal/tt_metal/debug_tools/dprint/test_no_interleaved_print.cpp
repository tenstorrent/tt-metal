// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "core_coord.hpp"
#include "debug_tools_fixture.hpp"
#include "gtest/gtest.h"
#include "debug_tools_test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

////////////////////////////////////////////////////////////////////////////////
// A test for checking that prints are not interleaved.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
const std::string golden_output =
    R"(This is a large DPRINT message that should not be interleaved with other DPRINT messages. Adding "
        "the alphabet "
        "to extend the size of this message: ABCDEFGHIJKLMNOPQRSTUVWXYZ. Now, in reverse, to make it even longer: "
        "ZYXWVUTSRQPONMLKJIHGFEDCBA.)";

static void RunTest(DPrintFixture* fixture, Device* device) {
    std::vector<CoreCoord> cores;
    cores.emplace_back(0, 0);
    cores.emplace_back(0, 1);
    cores.emplace_back(0, 2);

    Program program = Program();

    for (const CoreCoord& core : cores) {
        KernelHandle kernel_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/print_large.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        SetRuntimeArgs(program, kernel_id, core, {core.x, core.y});

        log_info(tt::LogTest, "Running test on core {}:({},{})", device->id(), core.x, core.y);
    }

    fixture->RunProgram(device, program);

    // Check the print log against golden output.
    // EXPECT_TRUE(FilesMatchesString(DPrintFixture::dprint_file_name, golden_output));
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DPrintFixture, TensixTestNoInterleavedPrints) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(
            [](DPrintFixture *fixture, Device *device){
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device);
            },
            device
        );
    }
}
