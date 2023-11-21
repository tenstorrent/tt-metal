// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "common/bfloat16.hpp"
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
    constexpr CoreCoord core = {0, 0}; // Print on first core only
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;
    Program program = Program();

    // Create a CB for testing TSLICE, dimensions are 32x32 bfloat16s
    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t buffer_size = 32*32*sizeof(bfloat16);
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
        buffer_size,
        {{src0_cb_index, tt::DataFormat::RawUInt32}}
    ).set_page_size(src0_cb_index, buffer_size);
    CircularBufferID cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Three different kernels to mirror typical usage and some previously
    // failing test cases, although all three kernels simply print.
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
    EXPECT_TRUE(
        FilesAreIdentical(
            CommandQueueWithDPrintFixture::dprint_file_name,
            "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/dprint/test_print_all_harts_golden.txt"
        )
    );
}
