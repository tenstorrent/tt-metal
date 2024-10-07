// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "dprint_fixture.hpp"
#include "gtest/gtest.h"
#include "impl/debug/dprint_server.hpp"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that the DPRINT server can be muted/unmuted.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

const std::string golden_output =
R"(Printing int from arg: 0
Printing int from arg: 2)";

static void RunTest(DPrintFixture* fixture, Device* device) {
    // Set up program
    auto program = CreateScopedProgram();

    // This tests prints only on a single core
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_one_int.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    // A lambda to run the program w/ a given test number (used in the printing).
    auto run_program = [&](uint32_t test_number) {
        SetRuntimeArgs(
            program,
            brisc_print_kernel_id,
            core,
            {test_number}
        );
        fixture->RunProgram(device, program);
    };

    // Run the program, prints should be enabled.
    run_program(0);

    // Disable the printing and run the program again.
    DprintServerSetMute(true);
    run_program(1);

    // Re-enable prints and run the program one more time.
    DprintServerSetMute(false);
    run_program(2);

    // Check the print log against golden output.
    EXPECT_TRUE(
        FilesMatchesString(
            DPrintFixture::dprint_file_name,
            golden_output
        )
    );
}

TEST_F(DPrintFixture, TestPrintMuting) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
