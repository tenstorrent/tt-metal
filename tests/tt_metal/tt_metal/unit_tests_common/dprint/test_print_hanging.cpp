// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_fixture.hpp"
#include "common/bfloat16.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that we can handle an invalid WAIT command.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

// Some machines will run this test on different physical cores, so wildcard the exact coordinates.
const std::string golden_output =
R"(DPRINT server timed out on Device *, worker core (x=*,y=*), riscv 4, waiting on a RAISE signal: 1
)";

static void RunTest(DPrintFixture* fixture, Device* device) {
    // Set up program
    auto program = CreateScopedProgram();

    // Run a kernel that just waits on a signal that never comes (BRISC only).
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    // Run the program, we expect it to throw on waiting for CQ to finish
try {
    fixture->RunProgram(device, program);
} catch (std::runtime_error& e) {
    const string expected = "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.";
    const string error = string(e.what());
    log_info(tt::LogTest, "Caught exception (one is expected in this test)");
    EXPECT_TRUE(error.find(expected) != string::npos);
}

    // Check the print log against golden output.
    EXPECT_TRUE(
        FilesMatchesString(
            DPrintFixture::dprint_file_name,
            golden_output
        )
    );
}

TEST_F(DPrintFixture, TestPrintHanging) {
    // Skip this test for slow dipatch for now. Due to how llrt currently sits below device, it's
    // tricky to check print server status from the finish loop for slow dispatch. Once issue #4363
    // is resolved, we should add a check for print server handing in slow dispatch as well.
    if (this->slow_dispatch_)
        GTEST_SKIP();

    // Since the dprint server gets killed from a timeout, only run on one device.
    this->RunTestOnDevice(RunTest, this->devices_[0]);
}
