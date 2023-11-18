// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "command_queue_fixture.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

////////////////////////////////////////////////////////////////////////////////
// A test for DPrint RAISE/WAIT between cores and riscs.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

TEST_F(CommandQueueWithDPrintFixture, TestPrintRaiseWait) {
    // This test is a fast dispatch test.
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }
try{
    // Device already set up by gtest fixture.
    const Device& device = this->device_;

    // Set up program and command queue
    CommandQueue& cq = *tt::tt_metal::detail::GLOBAL_CQ;
    Program program = Program();

    // Test runs on a 5x5 grid
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = {4, 4};

    // Two kernels - one for brisc and one for ncrisc. Nothing for triscs in
    // this test.
    KernelID brisc_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dprint_raise_wait_brisc.cpp",
        CoreRange{
            .start = xy_start,
            .end = xy_end
        },
        DataMovementConfig {
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default
        }
    );
    KernelID ncrisc_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dprint_raise_wait_ncrisc.cpp",
        CoreRange{
            .start = xy_start,
            .end = xy_end
        },
        DataMovementConfig {
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default
        }
    );

    // Write runtime args
    uint32_t multi_core = 1;
    for (uint32_t x = xy_start.x; x <= xy_end.x; x++) {
        for (uint32_t y = xy_start.y; y <= xy_end.y; y++) {
            const std::vector<uint32_t> brisc_rt_args = {
                x, y, multi_core, 15
            };
            SetRuntimeArgs(
                program,
                brisc_kernel_id,
                CoreCoord{x, y},
                brisc_rt_args
            );
            const std::vector<uint32_t> ncrisc_rt_args = {
                x, y, (uint32_t) xy_end.x+1, multi_core, 4, 2, 10
            };
            SetRuntimeArgs(
                program,
                ncrisc_kernel_id,
                CoreCoord{x, y},
                ncrisc_rt_args
            );
        }
    }


    // Run the program
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Since the program takes almost no time to run, wait a bit for the print
    // server to catch up.
    std::this_thread::sleep_for (std::chrono::seconds(1));

    // Check the print log against golden output.
    EXPECT_TRUE(
        FilesAreIdentical(
            CommandQueueWithDPrintFixture::dprint_file_name,
            "tests/tt_metal/tt_metal/unit_tests_fast_dispatch/dprint/test_raise_wait_golden.txt"
        )
    );
} catch (std::exception& e) {
    TT_THROW("Exception: {}", e.what());
}
}
