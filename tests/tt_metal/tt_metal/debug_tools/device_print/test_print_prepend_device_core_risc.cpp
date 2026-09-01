// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include <tt-metalium/device.hpp>
#include "gtest/gtest.h"
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

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
    // by machine. CoreCoord::str() formats as "x-y".
    const std::string& device_core_risc = std::to_string(mesh_device->get_device_ids()[0]) + ":?-?:" + risc + ": ";

    const std::string& output_line_all_riscs = device_core_risc + "Printing on a RISC.";
    golden_output.push_back(output_line_all_riscs);

    if (risc != "ER0" && risc != "ER") {
        const std::string& output_line_risc = device_core_risc + "Printing on " + risc + ".";
        golden_output.push_back(output_line_risc);
    }
}

constexpr const char* kPrintSimpleKernel = "tests/tt_metal/tt_metal/test_kernels/device_print/print_simple.cpp";

// The Tensix half of the test, on the Metal 2.0 host API: BRISC + NCRISC + compute on every core in
// `cores`. This is Gen1-only by construction -- the golden output below spells out the Gen1 RISC
// names (BR/NC/TR0/TR1/TR2), which have no Gen2 equivalent.
Program MakeTensixPrependProgram(distributed::MeshDevice& mesh_device, const CoreRange& cores) {
    const experimental::KernelSpecName kBrisc{"prepend_brisc"};
    const experimental::KernelSpecName kNcrisc{"prepend_ncrisc"};
    const experimental::KernelSpecName kCompute{"prepend_compute"};
    const auto source = std::filesystem::path{kPrintSimpleKernel};

    // Two dedicated-NOC DM kernels on one node must take distinct NOCs, or spec validation rejects
    // them (and the hardware would deadlock).
    experimental::ProgramSpec spec{
        .name = "dprint_prepend",
        .kernels =
            {experimental::KernelSpec{
                 .unique_id = kBrisc,
                 .source = source,
                 .num_threads = 1,
                 .hw_config =
                     experimental::DataMovementHardwareConfig{
                         .gen1_specific =
                             experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                                 .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0}},
             },
             experimental::KernelSpec{
                 .unique_id = kNcrisc,
                 .source = source,
                 .num_threads = 1,
                 .hw_config =
                     experimental::DataMovementHardwareConfig{
                         .gen1_specific =
                             experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                                 .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1}},
             },
             experimental::KernelSpec{
                 .unique_id = kCompute,
                 .source = source,
                 .num_threads = 1,
                 .hw_config = experimental::ComputeHardwareConfig{},
             }},
        .work_units = {experimental::WorkUnitSpec{
            .name = "main", .kernels = {kBrisc, kNcrisc, kCompute}, .target_nodes = experimental::NodeRange{cores}}},
    };
    return experimental::MakeProgramFromSpec(mesh_device, spec);
}

void RunTest(
    DevicePrintFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const bool add_active_eth_kernel = false) {
    std::vector<std::string> golden_output;

    CoreRange cores({0, 0}, {0, 1});
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->get_devices()[0];

    // The ethernet variant still needs the legacy API (Metal 2.0 places kernels on Tensix only), and
    // a Program cannot mix the two APIs -- so that variant keeps the whole program on the old path.
    Program program = add_active_eth_kernel ? Program() : MakeTensixPrependProgram(*mesh_device, cores);
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    if (add_active_eth_kernel) {
        CreateKernel(
            program_,
            kPrintSimpleKernel,
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        CreateKernel(
            program_,
            kPrintSimpleKernel,
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        CreateKernel(program_, kPrintSimpleKernel, cores, ComputeConfig{});
    }

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
        tt_metal::EthernetConfig config = {.noc = tt_metal::NOC::NOC_0, .processor = DataMovementProcessor::RISCV_0};
        eth_test_common::set_arch_specific_eth_config(config);
        CreateKernel(program_, kPrintSimpleKernel, crs, config);

        for ([[maybe_unused]] const CoreCoord& core : active_eth_cores) {
            if (tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
                    HalProgrammableCoreType::ACTIVE_ETH) > 1) {
                UpdateGoldenOutput(golden_output, mesh_device, "ER0");
            } else {
                UpdateGoldenOutput(golden_output, mesh_device, "ER");
            }
        }
    }

    fixture->RunProgram(mesh_device, workload);

    // Check the print log against golden output.
    EXPECT_TRUE(FileContainsAllStrings(fixture->dprint_file_name, golden_output));
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DevicePrintFixture, TensixTestPrintPrependDeviceCoreRisc) {
    if (MetalContext::instance().hal().get_arch() == tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Golden output is specific to the Gen1 RISC naming";
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, true);
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device);
            },
            mesh_device);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, false);
}

TEST_F(DevicePrintFixture, TensixActiveEthTestPrintPrependDeviceCoreRisc) {
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, true);
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->get_devices()[0]->get_active_ethernet_cores(true).empty()) {
            const auto device_id = mesh_device->get_device_ids()[0];
            log_info(tt::LogTest, "Skipping device {} due to no active ethernet cores...", device_id);
            continue;
        }
        this->RunTestOnDevice(
            [](DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device, true);
            },
            mesh_device);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, false);
}
