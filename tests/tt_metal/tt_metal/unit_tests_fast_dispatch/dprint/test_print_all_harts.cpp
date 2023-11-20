// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "llrt/tt_debug_print_server.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A simple test for checking DPRINTs from all harts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

TEST_F(CommandQueueWithDPrintFixture, TestPrintFromAllHarts) {
    bool pass = true;

    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    // Device already set up by gtest fixture.
    Device *device = this->device_;

    // Set up program and command queue
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;
    Program program = Program();

    // Three different kernels to mirror typical usage and some previously
    // failing test cases, although all three kernels simply print.
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    KernelHandle brisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/brisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default}
    );
    KernelHandle ncrisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/ncrisc_print.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default}
    );
    KernelHandle trisc_print_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/trisc_print.cpp",
        core,
        ComputeConfig{}
    );

    // Run the program
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Wait for the print server to catch up
    tt_await_debug_print_server();

    // Check that the expected print messages are in the log file
    vector<string> expected_prints({
        "Test Debug Print: Pack",
        "Test Debug Print: Unpack",
        "Test Debug Print: Math",
        "Test Debug Print: Data0",
        "Test Debug Print: Data1"
    });
    EXPECT_TRUE(
        FileContainsAllStrings(
            CommandQueueWithDPrintFixture::dprint_file_name,
            expected_prints
        )
    );
}
