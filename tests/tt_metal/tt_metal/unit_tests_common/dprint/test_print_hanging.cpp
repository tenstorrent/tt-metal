// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_fixture.hpp"
#include "common/bfloat16.hpp"
#include "impl/debug/dprint_server.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking that we can handle an invalid WAIT command.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

const std::string golden_output =
R"(DPRINT server timed out on core (1,1) riscv 4, waiting on a RAISE signal: 1
)";

TEST_F(DPrintFixture, TestPrintHanging) {
    // Device already set up by gtest fixture.
    Device *device = this->device_;

    // Set up program and command queue
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;
    Program program = Program();

    // Run a kernel that just waits on a signal that never comes (BRISC only).
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_hang.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );

    // Run the program, we expect it to throw on waiting for CQ to finish
    EnqueueProgram(cq, program, false);
try {
    Finish(cq);
    tt_await_debug_print_server();
} catch (std::runtime_error& e) {
    const string expected = "Command Queue could not finish: device hang due to unanswered DPRINT WAIT.";
    const string error = string(e.what());
    log_info(tt::LogTest, "Caught exception (one is expected in this test): {}", error);
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
