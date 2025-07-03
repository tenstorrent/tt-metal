// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <functional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "umd/device/types/arch.h"
#include "umd/device/types/xy_pair.h"

////////////////////////////////////////////////////////////////////////////////
// A test for printing from ethernet cores.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
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

void RunTest(
    DPrintFixture* fixture,
    IDevice* device,
    bool active,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0) {
    // Try printing on all ethernet cores on this device
    int count = 0;
    std::unordered_set<CoreCoord> test_cores;
    tt_metal::EthernetConfig config = {.noc = tt_metal::NOC::NOC_0, .processor = processor};
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
        KernelHandle erisc_kernel_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/erisc_print.cpp",
            core,
            config);

        // Run the program
        log_info(
            tt::LogTest,
            "Running print test on eth core {}:({},{}), {}",
            device->id(),
            core.x,
            core.y,
            processor
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
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}
}
}

TEST_F(DPrintFixture, ActiveEthTestPrint) {
    for (IDevice* device : this->devices_) {
        // Skip if no ethernet cores on this device
        if (device->get_active_ethernet_cores(true).size() == 0) {
            log_info(tt::LogTest, "Skipping device {} due to no ethernet cores...", device->id());
            continue;
        }
        this->RunTestOnDevice(
            [](DPrintFixture *fixture, IDevice* device){
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device, true);
            },
            device
        );
    }
}
TEST_F(DPrintFixture, IdleEthTestPrint) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    for (IDevice* device : this->devices_) {
        // Skip if no ethernet cores on this device
        if (device->get_inactive_ethernet_cores().size() == 0) {
            log_info(tt::LogTest, "Skipping device {} due to no ethernet cores...", device->id());
            continue;
        }
        this->RunTestOnDevice(
            [](DPrintFixture* fixture, IDevice* device) { CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device, false); },
            device);
        if (device->arch() == ARCH::BLACKHOLE) {
            this->RunTestOnDevice(
                [](DPrintFixture* fixture, IDevice* device) {
                    CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device, false, DataMovementProcessor::RISCV_1);
                },
                device);
        }
    }
}
