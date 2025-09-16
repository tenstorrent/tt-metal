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
#include "assert.hpp"
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

void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProcessorIdentifier processor) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::AddProgramToMeshWorkload(workload, {}, device_range);
    auto& program = workload.get_programs().at(device_range);
    auto device = mesh_device->get_devices()[0];

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
                    switch (processor.processor_type) {
                        case 0:
                            dm_config.processor = tt_metal::DataMovementProcessor::RISCV_0;
                            dm_config.noc = tt_metal::NOC::RISCV_0_default;
                            break;
                        case 1:
                            dm_config.processor = tt_metal::DataMovementProcessor::RISCV_1;
                            dm_config.noc = tt_metal::NOC::RISCV_1_default;
                            break;
                        default: TT_THROW("Unsupported DM processor type {}", processor.processor_type);
                    }
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

TEST_F(MeshWatcherFixture, TestWatcherRingBufferBrisc) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {TENSIX, DM, 0});
            },
            mesh_device);
    }
}

TEST_F(MeshWatcherFixture, TestWatcherRingBufferNCrisc) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {TENSIX, DM, 1});
            },
            mesh_device);
    }
}

TEST_F(MeshWatcherFixture, TestWatcherRingBufferTrisc0) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {TENSIX, COMPUTE, 0});
            },
            mesh_device);
    }
}

TEST_F(MeshWatcherFixture, TestWatcherRingBufferTrisc1) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {TENSIX, COMPUTE, 1});
            },
            mesh_device);
    }
}

TEST_F(MeshWatcherFixture, TestWatcherRingBufferTrisc2) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {TENSIX, COMPUTE, 2});
            },
            mesh_device);
    }
}

TEST_F(MeshWatcherFixture, TestWatcherRingBufferErisc) {
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {ACTIVE_ETH, DM, 0});
            },
            mesh_device);
    }
}

TEST_F(MeshWatcherFixture, TestWatcherRingBufferIErisc) {
    if (!this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {IDLE_ETH, DM, 0});
            },
            mesh_device);
    }
}

}  // namespace
