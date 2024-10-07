// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_fixture.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

////////////////////////////////////////////////////////////////////////////////
// A test for DPrint RAISE/WAIT between cores and riscs.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

const std::string golden_output =
R"(TestConstCharStrNC{0,0}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{0,0}
+++++++++++++++
TestConstCharStrNC{1,0}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{1,0}
+++++++++++++++
TestConstCharStrNC{2,0}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{2,0}
+++++++++++++++
TestConstCharStrNC{3,0}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{3,0}
+++++++++++++++
TestConstCharStrNC{4,0}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{4,0}
+++++++++++++++
TestConstCharStrNC{0,1}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{0,1}
+++++++++++++++
TestConstCharStrNC{1,1}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{1,1}
+++++++++++++++
TestConstCharStrNC{2,1}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{2,1}
+++++++++++++++
TestConstCharStrNC{3,1}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{3,1}
+++++++++++++++
TestConstCharStrNC{4,1}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{4,1}
+++++++++++++++
TestConstCharStrNC{0,2}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{0,2}
+++++++++++++++
TestConstCharStrNC{1,2}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{1,2}
+++++++++++++++
TestConstCharStrNC{2,2}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{2,2}
+++++++++++++++
TestConstCharStrNC{3,2}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{3,2}
+++++++++++++++
TestConstCharStrNC{4,2}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{4,2}
+++++++++++++++
TestConstCharStrNC{0,3}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{0,3}
+++++++++++++++
TestConstCharStrNC{1,3}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{1,3}
+++++++++++++++
TestConstCharStrNC{2,3}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{2,3}
+++++++++++++++
TestConstCharStrNC{3,3}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{3,3}
+++++++++++++++
TestConstCharStrNC{4,3}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{4,3}
+++++++++++++++
TestConstCharStrNC{0,4}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{0,4}
+++++++++++++++
TestConstCharStrNC{1,4}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{1,4}
+++++++++++++++
TestConstCharStrNC{2,4}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{2,4}
+++++++++++++++
TestConstCharStrNC{3,4}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{3,4}
+++++++++++++++
TestConstCharStrNC{4,4}
   2
0.1235
0.1200
0.1226
----------
TestStrBR{4,4}
+++++++++++++++)";

static void RunTest(DPrintFixture* fixture, Device* device) {
    // Set up program and command queue
    auto program = CreateScopedProgram();

    // Test runs on a 5x5 grid
    CoreCoord xy_start = {0, 0};
    CoreCoord xy_end = {4, 4};

    // Two kernels - one for brisc and one for ncrisc. Nothing for triscs in
    // this test.
    KernelHandle brisc_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dprint_raise_wait_brisc.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    KernelHandle ncrisc_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/dprint_raise_wait_ncrisc.cpp",
        CoreRange(xy_start, xy_end),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

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
    fixture->RunProgram(device, program);

    // Check the print log against golden output.
    EXPECT_TRUE(
        FilesMatchesString(
            DPrintFixture::dprint_file_name,
            golden_output
        )
    );
}

TEST_F(DPrintFixture, TestPrintRaiseWait) {
    for (Device* device : this->devices_) {
        this->RunTestOnDevice(RunTest, device);
    }
}
