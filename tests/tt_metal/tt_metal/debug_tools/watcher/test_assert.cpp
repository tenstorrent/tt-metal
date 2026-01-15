// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt_stl/assert.hpp>
#include "debug_tools_fixture.hpp"
#include "hal_types.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher asserts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {
static void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProgrammableCoreType programmable_core_type,
    HalProcessorClassType processor_class,
    int processor_id,
    dev_msgs::debug_assert_type_t assert_type = dev_msgs::DebugAssertTripped) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];

    // Depending on riscv type, choose one core to run the test on (since the test hangs the board).
    CoreCoord logical_core, virtual_core;
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            logical_core = {0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            if (device->get_active_ethernet_cores(true).empty()) {
                log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_active_ethernet_cores(true).begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            if (device->get_inactive_ethernet_cores().empty()) {
                log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_inactive_ethernet_cores().begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }
    log_info(LogTest, "Running test on device {} core {}...", device->id(), virtual_core.str());

    // Set up the kernel on the correct risc
    KernelHandle assert_kernel;
    std::string risc;
    switch (programmable_core_type) {
        case HalProgrammableCoreType::TENSIX:
            switch (processor_class) {
                case HalProcessorClassType::DM:
                    switch (processor_id) {
                        case 0:
                            assert_kernel = CreateKernel(
                                program_,
                                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                                logical_core,
                                DataMovementConfig{
                                    .processor = tt_metal::DataMovementProcessor::RISCV_0,
                                    .noc = tt_metal::NOC::RISCV_0_default});
                            risc = " brisc";
                            break;
                        case 1:
                            assert_kernel = CreateKernel(
                                program_,
                                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                                logical_core,
                                DataMovementConfig{
                                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                                    .noc = tt_metal::NOC::RISCV_1_default});
                            risc = "ncrisc";
                            break;
                        default: TT_THROW("Unsupported DM processor id {}", processor_id);
                    }
                    break;
                case HalProcessorClassType::COMPUTE:
                    TT_FATAL(
                        0 <= processor_id && processor_id < 3,
                        "processor_id {} must be 0, 1, or 2 for COMPUTE",
                        processor_id);
                    assert_kernel = CreateKernel(
                        program_,
                        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                        logical_core,
                        ComputeConfig{.defines = {{fmt::format("TRISC{}", processor_id), "1"}}});
                    risc = fmt::format("trisc{}", processor_id);
                    break;
            }
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.noc = tt_metal::NOC::NOC_0});
            risc = "erisc";
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});
            risc = "erisc";
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }

    // Write runtime args that should not trip an assert.
    const std::vector<uint32_t> safe_args = {3, 4, static_cast<uint32_t>(assert_type)};
    SetRuntimeArgs(program_, assert_kernel, logical_core, safe_args);

    // Run the kernel, don't expect an issue here.
    log_info(LogTest, "Running args that shouldn't assert...");
    fixture->RunProgram(mesh_device, workload, true);
    log_info(LogTest, "Args did not assert!");

    // Write runtime args that should trip an assert.
    const std::vector<uint32_t> unsafe_args = {3, 3, static_cast<uint32_t>(assert_type)};
    SetRuntimeArgs(program_, assert_kernel, logical_core, unsafe_args);

    // Run the kernel, expect an exit due to the assert.
    log_info(LogTest, "Running args that should assert...");
    fixture->RunProgram(mesh_device, workload);

    // We should be able to find the expected watcher error in the log as well,
    // expected error message depends on the risc we're running on and the assert type.
    const std::string kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp";
    std::string expected;
    if (assert_type == dev_msgs::DebugAssertTripped) {
        const uint32_t line_num = 66;
        expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} tripped an assert on line {}. "
            "Note that file name reporting is not yet implemented, and the reported line number for the assert may be "
            "from a different file. Current kernel: {}.",
            device->id(),
            (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) ? "acteth" : "worker",
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            risc,
            line_num,
            kernel);
    } else {
        std::string barrier;
        if (assert_type == dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped) {
            barrier = "NOC non-posted atomics flushed";
        } else if (assert_type == dev_msgs::DebugAssertNCriscNOCNonpostedWritesSentTripped) {
            barrier = "NOC non-posted writes sent";
        } else if (assert_type == dev_msgs::DebugAssertNCriscNOCPostedWritesSentTripped) {
            barrier = "NOC posted writes sent";
        } else if (assert_type == dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped) {
            barrier = "NOC reads flushed";
        }

        expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} detected an inter-kernel data race due to "
            "kernel completing with pending NOC transactions (missing {} barrier). Current kernel: "
            "{}.",
            device->id(),
            (programmable_core_type == HalProgrammableCoreType::ACTIVE_ETH) ? "acteth" : "worker",
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            risc,
            barrier,
            kernel);
    }

    std::string exception;
    do {
        exception = MetalContext::instance().watcher_server()->exception_message();
    } while (exception.empty());
    EXPECT_EQ(expected, MetalContext::instance().watcher_server()->exception_message());
}
}

// Test parameters structure
struct WatcherTestParams {
    std::string test_name;
    HalProgrammableCoreType core_type;
    HalProcessorClassType processor_class;
    int processor_id;
    dev_msgs::debug_assert_type_t assert_type = dev_msgs::DebugAssertTripped;
};

class WatcherAssertTest : public MeshWatcherFixture, public ::testing::WithParamInterface<WatcherTestParams> {};

TEST_P(WatcherAssertTest, TestWatcherAssert) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& params = GetParam();

    if (params.core_type == HalProgrammableCoreType::IDLE_ETH && !this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    if (this->slow_dispatch_) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTest(
                fixture,
                mesh_device,
                params.core_type,
                params.processor_class,
                params.processor_id,
                params.assert_type);
        },
        this->devices_[0]);
}

namespace {

using enum HalProgrammableCoreType;
using enum HalProcessorClassType;

INSTANTIATE_TEST_SUITE_P(
    WatcherAssertTests,
    WatcherAssertTest,
    ::testing::Values(
        WatcherTestParams{"Brisc", TENSIX, DM, 0},
        WatcherTestParams{"NCrisc", TENSIX, DM, 1},
        WatcherTestParams{"Trisc0", TENSIX, COMPUTE, 0},
        WatcherTestParams{"Trisc1", TENSIX, COMPUTE, 1},
        WatcherTestParams{"Trisc2", TENSIX, COMPUTE, 2},
        WatcherTestParams{"Erisc", ACTIVE_ETH, DM, 0},
        WatcherTestParams{"IErisc", IDLE_ETH, DM, 0}),
    [](const ::testing::TestParamInfo<WatcherTestParams>& info) { return info.param.test_name; });

INSTANTIATE_TEST_SUITE_P(
    WatcherNonDefaultAssertTests,
    WatcherAssertTest,
    ::testing::Values(
        WatcherTestParams{"Brisc", TENSIX, DM, 0, dev_msgs::DebugAssertTripped},
        WatcherTestParams{"NCrisc", TENSIX, DM, 1, dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped},
        WatcherTestParams{"Trisc0", TENSIX, COMPUTE, 0, dev_msgs::DebugAssertNCriscNOCNonpostedWritesSentTripped},
        WatcherTestParams{"Trisc1", TENSIX, COMPUTE, 1, dev_msgs::DebugAssertNCriscNOCPostedWritesSentTripped},
        WatcherTestParams{"Trisc2", TENSIX, COMPUTE, 2, dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped},
        WatcherTestParams{"Erisc", ACTIVE_ETH, DM, 0, dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped},
        WatcherTestParams{"IErisc", IDLE_ETH, DM, 0, dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped}),
    [](const ::testing::TestParamInfo<WatcherTestParams>& info) { return info.param.test_name; });

}  // namespace
