// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <functional>
#include <set>
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
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/utils.hpp>

////////////////////////////////////////////////////////////////////////////////
// A test for checking that prints are prepended with their corresponding device, core and RISC.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
void UpdateGoldenOutput(std::vector<std::string>& golden_output, const IDevice* device, const std::string& risc) {
    // Using wildcard characters in lieu of actual values for the virtual coordinates as virtual coordinates can vary
    // by machine
    const std::string& device_core_risc = std::to_string(device->id()) + ":(x=?,y=?):" + risc + ": ";

    const std::string& output_line_all_riscs = device_core_risc + "Printing on a RISC.";
    golden_output.push_back(output_line_all_riscs);

    if (risc != "ER") {
        const std::string& output_line_risc = device_core_risc + "Printing on " + risc + ".";
        golden_output.push_back(output_line_risc);
    }
}

void RunTest(DPrintFixture* fixture, IDevice* device, const bool add_active_eth_kernel = false) {
    std::vector<std::string> golden_output;

    CoreRange cores({0, 0}, {0, 1});
    Program program = Program();

    KernelHandle brisc_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle ncrisc_kernel_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle trisc_kernel_id =
        CreateKernel(program, "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp", cores, ComputeConfig{});

    for (const CoreCoord& core : cores) {
        UpdateGoldenOutput(golden_output, device, "BR");
        UpdateGoldenOutput(golden_output, device, "NC");
        UpdateGoldenOutput(golden_output, device, "TR0");
        UpdateGoldenOutput(golden_output, device, "TR1");
        UpdateGoldenOutput(golden_output, device, "TR2");
    }

    if (add_active_eth_kernel) {
        const std::unordered_set<CoreCoord>& active_eth_cores = device->get_active_ethernet_cores(true);
        CoreRangeSet crs(std::set<CoreRange>(active_eth_cores.begin(), active_eth_cores.end()));
        KernelHandle erisc_kernel_id = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp",
            crs,
            EthernetConfig{.noc = NOC::NOC_0});

        for (const CoreCoord& core : active_eth_cores) {
            UpdateGoldenOutput(golden_output, device, "ER");
        }
    }

    fixture->RunProgram(device, program);

    // Check the print log against golden output.
    EXPECT_TRUE(FileContainsAllStrings(DPrintFixture::dprint_file_name, golden_output));
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DPrintFixture, TensixTestPrintPrependDeviceCoreRisc) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, true);
    for (IDevice* device : this->devices_) {
        this->RunTestOnDevice(
            [](DPrintFixture* fixture, IDevice* device) { CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device); }, device);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, false);
}

TEST_F(DPrintFixture, TensixActiveEthTestPrintPrependDeviceCoreRisc) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, true);
    for (IDevice* device : this->devices_) {
        if (device->get_active_ethernet_cores(true).empty()) {
            log_info(tt::LogTest, "Skipping device {} due to no active ethernet cores...", device->id());
            continue;
        }
        this->RunTestOnDevice(
            [](DPrintFixture* fixture, IDevice* device) { CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, device, true); },
            device);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, false);
}
