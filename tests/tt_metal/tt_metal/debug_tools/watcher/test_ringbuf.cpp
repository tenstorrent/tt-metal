// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <functional>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt_stl/assert.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hal_types.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking debug ring buffer feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

std::vector<std::string> expected = {
    "debug_ring_buffer=",
    "[0x00270028,0x00260027,0x00250026,0x00240025,0x00230024,0x00220023,0x00210022,0x00200021,",
    " 0x001f0020,0x001e001f,0x001d001e,0x001c001d,0x001b001c,0x001a001b,0x0019001a,0x00180019,",
    " 0x00170018,0x00160017,0x00150016,0x00140015,0x00130014,0x00120013,0x00110012,0x00100011,",
    " 0x000f0010,0x000e000f,0x000d000e,0x000c000d,0x000b000c,0x000a000b,0x0009000a,0x00080009,",
    "]"
};

namespace {

struct WatcherRBTestParams {
    std::string name;
    HalProcessorIdentifier processor;
};

class WatcherRBTest : public MeshWatcherFixture,
                      public ::testing::WithParamInterface<WatcherRBTestParams> {};

void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const WatcherRBTestParams& params) {

    HalProcessorIdentifier processor = params.processor;

    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, {});
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    // Depending on riscv type, choose one core to run the test on
    // and set up the kernel on the correct risc
    CoreCoord logical_core, virtual_core;
    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX:
            logical_core = CoreCoord{0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            switch (processor.processor_class) {
                case HalProcessorClassType::DM: {
                    DataMovementConfig dm_config{};
                    dm_config.processor = static_cast<tt_metal::DataMovementProcessor>(processor.processor_type);
                    // NOC selection: DM1 uses NOC_1 on BH/WH, all DMs on quasar use NOC_0
                    dm_config.noc = (processor.processor_type ==
                                         enchantum::to_underlying(tt::tt_metal::DataMovementProcessor::RISCV_1) &&
                                     MetalContext::instance().hal().get_num_nocs() > 1)
                                        ? tt_metal::NOC::RISCV_1_default
                                        : tt_metal::NOC::NOC_0;
                    CreateKernel(
                        program,
                        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                        logical_core,
                        dm_config);
                    break;
                }
                case HalProcessorClassType::COMPUTE:
                    CreateKernel(
                        program,
                        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                        logical_core,
                        ComputeConfig{.defines = {{fmt::format("TRISC{}", processor.processor_type), "1"}}});
                    break;
            }
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            if (device->get_active_ethernet_cores(true).empty()) {
                log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_active_ethernet_cores(true).begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                EthernetConfig{.noc = tt_metal::NOC::NOC_0});
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            if (device->get_inactive_ethernet_cores().empty()) {
                log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_inactive_ethernet_cores().begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported core type");
    }
    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, virtual_core);

    // Run the program
    fixture->RunProgram(mesh_device, workload, true);

    log_info(tt::LogTest, "Checking file: {}", fixture->log_file_name);

    // Check log
    EXPECT_TRUE(
        FileContainsAllStringsInOrder(
            fixture->log_file_name,
            expected
        )
    );
}

using enum HalProgrammableCoreType;
using enum HalProcessorClassType;

// Single parameterized test
TEST_P(WatcherRBTest, TestWatcherRingBuffer) {
    const auto& params = GetParam();

    // Skip if processor type is not available on this architecture
    const auto& hal = MetalContext::instance().hal();
    uint32_t core_type_index = hal.get_programmable_core_type_index(params.processor.core_type);
    uint32_t available_processors = hal.get_processor_types_count(
        core_type_index, static_cast<uint32_t>(params.processor.processor_class));
    if (params.processor.processor_type >= available_processors) {
        log_info(tt::LogTest, "Test {} requires processor type {} but only {} available.",
                 params.name, params.processor.processor_type, available_processors);
        GTEST_SKIP();
    }

    // Skip idle ethernet test if not slow dispatch
    if (params.processor.core_type == HalProgrammableCoreType::IDLE_ETH && !this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }

    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, params);
            },
            mesh_device);
    }
}

// Instantiate all test cases
// Tests are automatically skipped if the processor type is not available on the current architecture
INSTANTIATE_TEST_SUITE_P(
    WatcherRingBufferTests,
    WatcherRBTest,
    ::testing::Values(
        // DM processors
        WatcherRBTestParams{"Brisc", {TENSIX, DM, 0}},
        WatcherRBTestParams{"NCrisc", {TENSIX, DM, 1}},
        // DM2 to DM7 only run on Quasar
        WatcherRBTestParams{"DM2", {TENSIX, DM, 2}},
        WatcherRBTestParams{"DM3", {TENSIX, DM, 3}},
        WatcherRBTestParams{"DM4", {TENSIX, DM, 4}},
        WatcherRBTestParams{"DM5", {TENSIX, DM, 5}},
        WatcherRBTestParams{"DM6", {TENSIX, DM, 6}},
        WatcherRBTestParams{"DM7", {TENSIX, DM, 7}},
        // Compute processors
        WatcherRBTestParams{"Trisc0", {TENSIX, COMPUTE, 0}},
        WatcherRBTestParams{"Trisc1", {TENSIX, COMPUTE, 1}},
        WatcherRBTestParams{"Trisc2", {TENSIX, COMPUTE, 2}},
        WatcherRBTestParams{"Trisc3", {TENSIX, COMPUTE, 3}}, //Trisc3 only Runs on Quasar
        // Ethernet processors
        WatcherRBTestParams{"Erisc", {ACTIVE_ETH, DM, 0}},
        WatcherRBTestParams{"IErisc", {IDLE_ETH, DM, 0}}
    ),
    [](const ::testing::TestParamInfo<WatcherRBTestParams>& info) {
        return info.param.name;  // Use name as test suffix
    }
);

}  // namespace
