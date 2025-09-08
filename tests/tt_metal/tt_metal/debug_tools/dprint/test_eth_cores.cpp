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

#include <tt-metalium/distributed.hpp>
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
Pointer:
123
456
789
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
    DPrintMeshFixture* fixture,
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    bool active,
    DataMovementProcessor processor = DataMovementProcessor::RISCV_0) {
    auto device = mesh_device->get_devices()[0];
    // Try printing on all ethernet cores on this device
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
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        Program program = Program();
        distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
        auto& program_ = workload.get_programs().at(device_range);

        // Create the kernel
        CreateKernel(program_, "tests/tt_metal/tt_metal/test_kernels/misc/erisc_print.cpp", core, config);

        // Run the program
        log_info(
            tt::LogTest,
            "Running print test on eth core {}:({},{}), {}",
            device->id(),
            core.x,
            core.y,
            processor
        );
        fixture->RunProgram(mesh_device, workload);

        // Check the print log against golden output.
        EXPECT_TRUE(FilesMatchesString(DPrintMeshFixture::dprint_file_name, golden_output));

        // Clear the log file for the next core's test
        MetalContext::instance().dprint_server()->clear_log_file();
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

TEST_F(DPrintMeshFixture, ActiveEthTestPrint) {
    for (auto& mesh_device : this->devices_) {
        auto device = mesh_device->get_devices()[0];
        // Skip if no ethernet cores on this device
        if (device->get_active_ethernet_cores(true).size() == 0) {
            log_info(tt::LogTest, "Skipping device {} due to no ethernet cores...", device->id());
            continue;
        }

        const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(
            tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH);
        for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
            log_info(tt::LogTest, "Test active ethernet DM{}", erisc_idx);
            DataMovementProcessor dm_processor = static_cast<DataMovementProcessor>(erisc_idx);
            this->RunTestOnDevice(
                [=](DPrintMeshFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device) {
                    CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device, true, dm_processor);
                },
                mesh_device);
        }
    }
}
TEST_F(DPrintMeshFixture, IdleEthTestPrint) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    for (auto& mesh_device : this->devices_) {
        auto device = mesh_device->get_devices()[0];
        // Skip if no ethernet cores on this device
        if (device->get_inactive_ethernet_cores().size() == 0) {
            log_info(tt::LogTest, "Skipping device {} due to no ethernet cores...", device->id());
            continue;
        }
        const auto erisc_count = tt::tt_metal::MetalContext::instance().hal().get_processor_classes_count(
            tt::tt_metal::HalProgrammableCoreType::IDLE_ETH);
        for (uint32_t erisc_idx = 0; erisc_idx < erisc_count; erisc_idx++) {
            log_info(tt::LogTest, "Test idle ethernet DM{}", erisc_idx);
            DataMovementProcessor dm_processor = static_cast<DataMovementProcessor>(erisc_idx);

            this->RunTestOnDevice(
                [=](DPrintMeshFixture* fixture, std::shared_ptr<distributed::MeshDevice> mesh_device) {
                    CMAKE_UNIQUE_NAMESPACE::RunTest(fixture, mesh_device, false, dm_processor);
                },
                mesh_device);
        }
    }
}
