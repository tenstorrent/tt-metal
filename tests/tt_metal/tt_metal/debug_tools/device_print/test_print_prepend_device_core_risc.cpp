// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <functional>
#include <map>
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
#include <tt-metalium/hal_types.hpp>
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"
#include "distributed/mesh_device_impl.hpp"

////////////////////////////////////////////////////////////////////////////////
// A test for checking that prints are prepended with their corresponding device, core and RISC.
////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
// Adds the lines `risc` on `core` prints: print_simple.cpp's plain print, then, on a Tensix RISC, the variant that
// names it. The key is the line prefix, which uses logical core coordinates (CoreCoord::str() formats as "x-y").
void AddExpectedLines(
    std::map<std::string, std::vector<std::string>>& expected,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    const std::string& risc) {
    auto& lines = expected[fmt::format("{}:{}:{}", mesh_device->get_device_ids()[0], core.str(), risc)];
    lines.push_back("Printing on a RISC.");
    if (risc != "ER0" && risc != "ER") {
        lines.push_back("Printing on " + risc + ".");
    }
}

constexpr const char* kPrintSimpleKernel = "tests/tt_metal/tt_metal/test_kernels/device_print/print_simple.cpp";

// The Tensix half of the test, on the Metal 2.0 host API: BRISC + NCRISC + compute on every core in
// `cores`. This is Gen1-only by construction -- the golden output below spells out the Gen1 RISC
// names (BR/NC/TR0/TR1/TR2), which have no Gen2 equivalent.
Program MakeTensixPrependProgram(distributed::MeshDevice& mesh_device, const CoreRange& cores) {
    const KernelSpecName kBrisc{"prepend_brisc"};
    const KernelSpecName kNcrisc{"prepend_ncrisc"};
    const KernelSpecName kCompute{"prepend_compute"};
    const auto source = std::filesystem::path{kPrintSimpleKernel};

    // Two dedicated-NOC DM kernels on one node must take distinct NOCs, or spec validation rejects
    // them (and the hardware would deadlock).
    ProgramSpec spec{
        .name = "dprint_prepend",
        .kernels =
            {KernelSpec{
                 .unique_id = kBrisc,
                 .source = source,
                 .num_threads = 1,
                 .hw_config =
                     DataMovementHardwareConfig{
                         .config_1xx =
                             DataMovementHardwareConfig::DataMovement1XXConfig{
                                 .processor = DataMovementProcessor::RISCV_0,
                                 .noc = NOC::NOC_0,
                             },
                     },
             },
             KernelSpec{
                 .unique_id = kNcrisc,
                 .source = source,
                 .num_threads = 1,
                 .hw_config =
                     DataMovementHardwareConfig{
                         .config_1xx =
                             DataMovementHardwareConfig::DataMovement1XXConfig{
                                 .processor = DataMovementProcessor::RISCV_1,
                                 .noc = NOC::NOC_1,
                             },
                     },
             },
             KernelSpec{
                 .unique_id = kCompute,
                 .source = source,
                 .num_threads = 1,
                 .hw_config = ComputeHardwareConfig{},
             }},
        .work_units = {WorkUnitSpec{
            .name = "main", .kernels = {kBrisc, kNcrisc, kCompute}, .target_nodes = NodeRange{cores}}},
    };
    return tt_metal::Program(mesh_device, spec);
}

void RunTest(
    DevicePrintFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const bool add_active_eth_kernel = false) {
    std::map<std::string, std::vector<std::string>> expected;

    CoreRange cores({0, 0}, {0, 1});
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto* device = mesh_device->impl().get_devices()[0];

    // The ethernet variant still needs the legacy API (Metal 2.0 places kernels on Tensix only), and
    // a Program cannot mix the two APIs -- so that variant keeps the whole program on the old path.
    Program program =
        add_active_eth_kernel ? CreateProgram(*mesh_device) : MakeTensixPrependProgram(*mesh_device, cores);
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

    for (const CoreCoord& core : cores) {
        for (const char* risc : {"BR", "NC", "TR0", "TR1", "TR2"}) {
            AddExpectedLines(expected, mesh_device, core, risc);
        }
    }

    if (add_active_eth_kernel) {
        const std::unordered_set<CoreCoord>& active_eth_cores = device->get_active_ethernet_cores(true);
        CoreRangeSet crs(std::set<CoreRange>(active_eth_cores.begin(), active_eth_cores.end()));
        tt_metal::EthernetConfig config = {.noc = tt_metal::NOC::NOC_0, .processor = DataMovementProcessor::RISCV_0};
        eth_test_common::set_arch_specific_eth_config(config);
        CreateKernel(program_, kPrintSimpleKernel, crs, config);

        const bool two_eriscs = tt::tt_metal::MetalContext::instance().hal().get_num_risc_processors(
                                    HalProgrammableCoreType::ACTIVE_ETH) > 1;
        for (const CoreCoord& core : active_eth_cores) {
            AddExpectedLines(expected, mesh_device, core, two_eriscs ? "ER0" : "ER");
        }
    }

    fixture->RunProgram(mesh_device, workload);

    // Lines from different RISCs interleave in no defined order, so each RISC's lines are checked on their own.
    EXPECT_TRUE(FileMatchesLinesPerRisc(fixture->dprint_file_name, expected));
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
    if (not this->slow_dispatch_) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, true);
    bool ran = false;
    for (auto& mesh_device : this->devices_) {
        if (mesh_device->impl().get_devices()[0]->get_active_ethernet_cores(true).empty()) {
            const auto device_id = mesh_device->get_device_ids()[0];
            log_info(tt::LogTest, "Skipping device {} due to no active ethernet cores...", device_id);
            continue;
        }
        ran = true;
        this->RunTestOnDevice(
            [](DevicePrintFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device, true);
            },
            mesh_device);
    }
    tt::tt_metal::MetalContext::instance().rtoptions().set_feature_prepend_device_core_risc(
        tt::llrt::RunTimeDebugFeatureDprint, false);
    if (!ran) {
        GTEST_SKIP() << "No device has an active ethernet core";
    }
}
