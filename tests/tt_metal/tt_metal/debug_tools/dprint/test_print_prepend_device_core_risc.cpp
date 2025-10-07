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

#include <tt-metalium/distributed.hpp>
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
void UpdateGoldenOutput(
    std::vector<std::string>& golden_output,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const std::string& risc) {
    // Using wildcard characters in lieu of actual values for the virtual coordinates as virtual coordinates can vary
    // by machine
    const std::string& device_core_risc =
        std::to_string(mesh_device->get_devices()[0]->id()) + ":(x=?,y=?):" + risc + ": ";

    const std::string& output_line_all_riscs = device_core_risc + "Printing on a RISC.";
    golden_output.push_back(output_line_all_riscs);

    if (risc != "ER") {
        const std::string& output_line_risc = device_core_risc + "Printing on " + risc + ".";
        golden_output.push_back(output_line_risc);
    }
}

void RunTest(
    DPrintMeshFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const bool add_active_eth_kernel = false) {
    std::vector<std::string> golden_output;

    CoreRange cores({0, 0}, {0, 1});
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    auto& program_ = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    CreateKernel(program_, "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp", cores, ComputeConfig{});

    for ([[maybe_unused]] const CoreCoord& core : cores) {
        UpdateGoldenOutput(golden_output, mesh_device, "BR");
        UpdateGoldenOutput(golden_output, mesh_device, "NC");
        UpdateGoldenOutput(golden_output, mesh_device, "TR0");
        UpdateGoldenOutput(golden_output, mesh_device, "TR1");
        UpdateGoldenOutput(golden_output, mesh_device, "TR2");
    }

    if (add_active_eth_kernel) {
        const std::unordered_set<CoreCoord>& active_eth_cores = device->get_active_ethernet_cores(true);
        CoreRangeSet crs(std::set<CoreRange>(active_eth_cores.begin(), active_eth_cores.end()));
        CreateKernel(
            program_,
            "tests/tt_metal/tt_metal/test_kernels/misc/print_simple.cpp",
            crs,
            EthernetConfig{.noc = NOC::NOC_0});

        for ([[maybe_unused]] const CoreCoord& core : active_eth_cores) {
            UpdateGoldenOutput(golden_output, mesh_device, "ER");
        }
    }

    fixture->RunProgram(mesh_device, workload);

    // Check the print log against golden output.
    EXPECT_TRUE(FileContainsAllStrings(DPrintMeshFixture::dprint_file_name, golden_output));
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DPrintMeshFixture, TensixTestPrintPrependDeviceCoreRisc) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, true);
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device);
            },
            mesh_device);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, false);
}

TEST_F(DPrintMeshFixture, TensixActiveEthTestPrintPrependDeviceCoreRisc) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, true);
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->get_devices()[0]->get_active_ethernet_cores(true).empty()) {
            log_info(
                tt::LogTest,
                "Skipping device {} due to no active ethernet cores...",
                mesh_device->get_devices()[0]->id());
            continue;
        }
        this->RunTestOnDevice(
            [](DPrintMeshFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device, true);
            },
            mesh_device);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, false);
}
