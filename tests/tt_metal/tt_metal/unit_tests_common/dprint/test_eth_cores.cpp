// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint_fixture.hpp"
#include "gtest/gtest.h"
#include "test_utils.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

////////////////////////////////////////////////////////////////////////////////
// A test for printing from ethernet cores.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

const std::string golden_output =
R"(Test Debug Print: ERISC
Basic Types:
101-1.618@0.122559
e5551234569123456789
-17-343-44444-5123456789
SETPRECISION/FIXED/DEFAULTFLOAT:
3.1416
3.14159012
3.14159
3.141590118
SETW:
    123456123456  ab
HEX/OCT/DEC:
1e240361100123456)";

static void RunTest(DPrintFixture* fixture, Device* device, bool active) {
    // Try printing on all ethernet cores on this device
    int count = 0;
    std::unordered_set<CoreCoord> test_cores;
    tt_metal::EthernetConfig config = {.noc = tt_metal::NOC::NOC_0};
    if (active) {
        test_cores = device->get_active_ethernet_cores(true);
        config.eth_mode = Eth::SENDER;
    } else {
        test_cores = device->get_inactive_ethernet_cores();
        config.eth_mode = Eth::IDLE;
    }
    for (const auto& core : test_cores) {
        // Set up program and command queue
        Program program = Program();

        // Create the kernel
        // TODO: When #6424 is fixed combine these kernels again.
        KernelHandle erisc_kernel_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/erisc_print.cpp",
            core,
            config);

        // Run the program
        log_info(
            tt::LogTest,
            "Running print test on eth core {}:({},{})",
            device->id(),
            core.x,
            core.y
        );
        fixture->RunProgram(device, program);

        // Check the print log against golden output.
        EXPECT_TRUE(
            FilesMatchesString(
                DPrintFixture::dprint_file_name,
                golden_output
            )
        );

        // Clear the log file for the next core's test
        tt::DPrintServerClearLogFile();
    }
}

TEST_F(DPrintFixture, TestPrintEthCores) {
    for (Device* device : this->devices_) {
        // Skip if no ethernet cores on this device
        if (device->get_active_ethernet_cores(true).size() == 0) {
            log_info(tt::LogTest, "Skipping device {} due to no ethernet cores...", device->id());
            continue;
        }
        this->RunTestOnDevice(
            [](DPrintFixture *fixture, Device *device){
                RunTest(fixture, device, true);
            },
            device
        );
    }
}
TEST_F(DPrintFixture, TestPrintIEthCores) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    for (Device* device : this->devices_) {
        // Skip if no ethernet cores on this device
        if (device->get_inactive_ethernet_cores().size() == 0) {
            log_info(tt::LogTest, "Skipping device {} due to no ethernet cores...", device->id());
            continue;
        }
        this->RunTestOnDevice(
            [](DPrintFixture *fixture, Device *device){
                RunTest(fixture, device, false);
            },
            device
        );
    }
}
