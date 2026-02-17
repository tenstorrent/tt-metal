// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <regex>
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
#include "impl/kernels/kernel.hpp"
#include <tt-metalium/experimental/host_api.hpp>

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking watcher asserts.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace CMAKE_UNIQUE_NAMESPACE {
static std::string regex_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        if (std::string("\\^$.|?*+()[]{}").find(c) != std::string::npos) {
            result += '\\';
        }
        result += c;
    }
    return result;
}

static void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProcessorIdentifier processor,
    dev_msgs::debug_assert_type_t assert_type = dev_msgs::DebugAssertTripped) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = Program();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;

    // Depending on riscv type, choose one core to run the test on
    // and set up the kernel on the correct risc
    CoreCoord logical_core, virtual_core;
    // Set up the kernel on the correct risc
    KernelHandle assert_kernel;
    auto procesor_idx =
        hal.get_processor_index(processor.core_type, processor.processor_class, processor.processor_type);
    std::string risc = hal.get_processor_class_name(processor.core_type, procesor_idx, false).c_str();
    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX:
            logical_core = {0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            switch (processor.processor_class) {
                case HalProcessorClassType::DM:
                    if (is_quasar) {
                        assert_kernel = experimental::quasar::CreateKernel(
                            program_,
                            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                            logical_core,
                            experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 8});
                    } else {
                        DataMovementConfig dm_config{};
                        dm_config.processor = static_cast<tt_metal::DataMovementProcessor>(processor.processor_type);
                        dm_config.noc = (processor.processor_type ==
                                         enchantum::to_underlying(tt::tt_metal::DataMovementProcessor::RISCV_1))
                                            ? tt_metal::NOC::RISCV_1_default
                                            : tt_metal::NOC::RISCV_0_default;
                        assert_kernel = CreateKernel(
                            program_,
                            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                            logical_core,
                            dm_config);
                    }
                    break;
                case HalProcessorClassType::COMPUTE:
                    // TODO: Watcher features are temporarily skipped on Quasar until basic runtime bring-up is complete
                    if (is_quasar && processor.processor_type >= 3) {
                        return;
                    }
                    TT_FATAL(
                        0 <= processor.processor_type && processor.processor_type < 3,
                        "processor_id {} must be 0, 1, or 2 for COMPUTE",
                        processor.processor_type);
                    assert_kernel = CreateKernel(
                        program_,
                        "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
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
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.noc = tt_metal::NOC::NOC_0});
            risc = "erisc";
            break;
        case HalProgrammableCoreType::IDLE_ETH:
            if (device->get_inactive_ethernet_cores().empty()) {
                log_info(LogTest, "Skipping this test since device has no inactive ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_inactive_ethernet_cores().begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            assert_kernel = CreateKernel(
                program_,
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp",
                logical_core,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0});
            risc = "erisc";
            break;
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }
    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, virtual_core);

    // // Write runtime args that should not trip an assert.
    // const std::vector<uint32_t> safe_args = {3, 4, static_cast<uint32_t>(assert_type)};
    // SetRuntimeArgs(program_, assert_kernel, logical_core, safe_args);

    // // Run the kernel, don't expect an issue here.
    // log_info(LogTest, "Running args that shouldn't assert...");
    // fixture->RunProgram(mesh_device, workload, true);
    // log_info(LogTest, "Args did not assert!");

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
        const uint32_t line_num = 58;
        expected = fmt::format(
            "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} tripped an assert on line {}. "
            "Note that file name reporting is not yet implemented, and the reported line number for the assert may be "
            "from a different file. Current kernel: {}.",
            device->id(),
            (processor.core_type == HalProgrammableCoreType::ACTIVE_ETH) ? "acteth" : "worker",
            logical_core.x,
            logical_core.y,
            virtual_core.x,
            virtual_core.y,
            risc);
        std::string after_line = fmt::format(
            ". Note that file name reporting is not yet implemented, and the reported line number for the assert may "
            "be from a different file. Current kernel: {}.",
            kernel);
        expected = regex_escape(before_line) + "\\d+" + regex_escape(after_line);
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
            (processor.core_type == HalProgrammableCoreType::ACTIVE_ETH) ? "acteth" : "worker",
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
    if (assert_type == dev_msgs::DebugAssertTripped) {
        EXPECT_TRUE(std::regex_match(exception, std::regex(expected)))
            << "Expected pattern: " << expected << "\nActual: " << exception;
    } else {
        EXPECT_EQ(expected, exception);
    }
}
}

// Test parameters structure
struct WatcherTestParams {
    std::string test_name;
    HalProcessorIdentifier processor;
    dev_msgs::debug_assert_type_t assert_type = dev_msgs::DebugAssertTripped;
};

class WatcherAssertTest : public MeshWatcherFixture, public ::testing::WithParamInterface<WatcherTestParams> {};

TEST_P(WatcherAssertTest, TestWatcherAssert) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    const auto& params = GetParam();

    // Skip if processor type is not available on this architecture
    const auto& hal = MetalContext::instance().hal();
    uint32_t core_type_index = hal.get_programmable_core_type_index(params.processor.core_type);
    uint32_t available_processors =
        hal.get_processor_types_count(core_type_index, static_cast<uint32_t>(params.processor.processor_class));
    if (params.processor.processor_type >= available_processors) {
        log_info(
            tt::LogTest,
            "Test {} requires processor type {} but only {} available.",
            params.test_name,
            params.processor.processor_type,
            available_processors);
        GTEST_SKIP();
    }

    if (params.processor.core_type == HalProgrammableCoreType::IDLE_ETH && !this->IsSlowDispatch()) {
        log_info(tt::LogTest, "FD-on-idle-eth not supported.");
        GTEST_SKIP();
    }
    if (this->slow_dispatch_ && (tt::tt_metal::MetalContext::instance().hal().get_arch() != tt::ARCH::QUASAR)) {
        GTEST_SKIP();
    }
    this->RunTestOnDevice(
        [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
            RunTest(fixture, mesh_device, params.processor, params.assert_type);
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
        WatcherTestParams{"Brisc", {TENSIX, DM, 0}},
        WatcherTestParams{"NCrisc", {TENSIX, DM, 1}},
        // DM2 to DM7 only run on Quasar
        WatcherTestParams{"DM2", {TENSIX, DM, 2}},
        WatcherTestParams{"DM3", {TENSIX, DM, 3}},
        WatcherTestParams{"DM4", {TENSIX, DM, 4}},
        WatcherTestParams{"DM5", {TENSIX, DM, 5}},
        WatcherTestParams{"DM6", {TENSIX, DM, 6}},
        WatcherTestParams{"DM7", {TENSIX, DM, 7}},
        WatcherTestParams{"Trisc0", {TENSIX, COMPUTE, 0}},
        WatcherTestParams{"Trisc1", {TENSIX, COMPUTE, 1}},
        WatcherTestParams{"Trisc2", {TENSIX, COMPUTE, 2}},
        WatcherTestParams{"Trisc3", {TENSIX, COMPUTE, 3}},  // Trisc3 only Runs on Quasar
        WatcherTestParams{"Erisc", {ACTIVE_ETH, DM, 0}},
        WatcherTestParams{"IErisc", {IDLE_ETH, DM, 0}}),
    [](const ::testing::TestParamInfo<WatcherTestParams>& info) { return info.param.test_name; });

INSTANTIATE_TEST_SUITE_P(
    WatcherNonDefaultAssertTests,
    WatcherAssertTest,
    ::testing::Values(
        WatcherTestParams{"Brisc", {TENSIX, DM, 0}, dev_msgs::DebugAssertTripped},
        WatcherTestParams{"NCrisc", {TENSIX, DM, 1}, dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped},
        WatcherTestParams{"Trisc0", {TENSIX, COMPUTE, 0}, dev_msgs::DebugAssertNCriscNOCNonpostedWritesSentTripped},
        WatcherTestParams{"Trisc1", {TENSIX, COMPUTE, 1}, dev_msgs::DebugAssertNCriscNOCPostedWritesSentTripped},
        WatcherTestParams{"Trisc2", {TENSIX, COMPUTE, 2}, dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped},
        WatcherTestParams{"Erisc", {ACTIVE_ETH, DM, 0}, dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped},
        WatcherTestParams{"IErisc", {IDLE_ETH, DM, 0}, dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped}),
    [](const ::testing::TestParamInfo<WatcherTestParams>& info) { return info.param.test_name; });

}  // namespace
