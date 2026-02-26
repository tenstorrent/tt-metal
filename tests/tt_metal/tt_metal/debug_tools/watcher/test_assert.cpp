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
#include <thread>
#include <chrono>

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
#include "impl/debug/debug_helpers.hpp"

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
    const std::string kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_asserts.cpp";

    // Depending on riscv type, choose one core to run the test on (since the test hangs the board).
    CoreCoord logical_core, virtual_core;
    // Set up the kernel on the correct risc
    KernelHandle assert_kernel;
    auto processor_idx =
        hal.get_processor_index(processor.core_type, processor.processor_class, processor.processor_type);
    std::string risc = hal.get_processor_class_name(processor.core_type, processor_idx, false);
    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX:
            logical_core = {0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            switch (processor.processor_class) {
                case HalProcessorClassType::DM:
                    if (is_quasar) {
                        // On Quasar, kernel runs on all 8 DMs but only dm_id executes the test;
                        // others exit early. This lets us verify assert works on each DM individually
                        uint32_t dm_id = static_cast<uint32_t>(processor.processor_type);
                        assert_kernel = tt::tt_metal::experimental::quasar::CreateKernel(
                            program_,
                            kernel,
                            logical_core,
                            tt::tt_metal::experimental::quasar::QuasarDataMovementConfig{
                                .num_processors_per_cluster = 8, .compile_args = {dm_id}});
                    } else {
                        DataMovementConfig dm_config{};
                        dm_config.processor = static_cast<tt_metal::DataMovementProcessor>(processor.processor_type);
                        dm_config.noc = (processor.processor_type ==
                                         enchantum::to_underlying(tt::tt_metal::DataMovementProcessor::RISCV_1))
                                            ? tt_metal::NOC::RISCV_1_default
                                            : tt_metal::NOC::RISCV_0_default;
                        assert_kernel = CreateKernel(program_, kernel, logical_core, dm_config);
                    }
                    break;
                case HalProcessorClassType::COMPUTE:
                    // TODO: Watcher features are temporarily skipped on Quasar until basic runtime bring-up is complete
                    if (is_quasar) {
                        GTEST_SKIP() << "Compute kernel watcher tests skipped on Quasar until TRISC runtime bring-up "
                                        "is complete";
                    }
                    assert_kernel = CreateKernel(
                        program_,
                        kernel,
                        logical_core,
                        ComputeConfig{.defines = {{fmt::format("TRISC{}", processor.processor_type), "1"}}});
                    break;
                default: TT_THROW("Unsupported processor class type for TENSIX");
            }
            break;
        case HalProgrammableCoreType::ACTIVE_ETH:
        case HalProgrammableCoreType::IDLE_ETH: {
            bool is_active = (processor.core_type == HalProgrammableCoreType::ACTIVE_ETH);
            auto eth_cores =
                is_active ? device->get_active_ethernet_cores(true) : device->get_inactive_ethernet_cores();
            if (eth_cores.empty()) {
                log_info(LogTest, "Skipping: device has no {} ethernet cores.", is_active ? "active" : "inactive");
                GTEST_SKIP();
            }
            logical_core = *eth_cores.begin();
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            EthernetConfig eth_config{.noc = tt_metal::NOC::NOC_0};
            if (!is_active) {
                eth_config.eth_mode = Eth::IDLE;
            }
            assert_kernel = CreateKernel(program_, kernel, logical_core, eth_config);
            // TODO: replace string literals with hal.get_processor_class_name() after
            // unifying all tests + watcher_device_reader::get_riscv_name() with same method
            risc = is_active ? "erisc" : "ierisc";
            break;
        }
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported programmable core type");
    }
    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, virtual_core);

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

    // Wait for watcher to catch the assert with a timeout of 5s
    std::string exception;
    constexpr auto timeout = std::chrono::milliseconds(5000);
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < timeout) {
        exception = MetalContext::instance().watcher_server()->exception_message();
        if (!exception.empty()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    ASSERT_FALSE(exception.empty()) << "Timeout (" << timeout.count() << "ms) waiting for watcher exception.\n"
                                    << "Expected assert type: " << static_cast<int>(assert_type) << " on " << risc;

    // We should be able to find the expected watcher error in the log as well,
    // expected error message depends on the risc we're running on and the assert type.
    // TODO: replace below code snippet with helper after unifying all tests + watcher_device_reader
    std::string core_str;
    switch (processor.core_type) {
        case HalProgrammableCoreType::ACTIVE_ETH: core_str = "acteth"; break;
        case HalProgrammableCoreType::IDLE_ETH: core_str = "idleth"; break;
        default: core_str = "worker";
    }

    // Don't hardcode line number, the ASSERT location in watcher_asserts.cpp kernel
    // can shift as code changes. Use regex to match any line number for DebugAssertTripped
    // (get_debug_assert_message defaults to line 0, which we replace with \d+ below)
    const std::string msg = get_debug_assert_message(assert_type);
    ASSERT_FALSE(msg.empty()) << "Unhandled assert type " << static_cast<int>(assert_type);

    std::string expected = fmt::format(
        "Device {} {} core(x={:2},y={:2}) virtual(x={:2},y={:2}): {} {} Current kernel: {}.",
        device->id(),
        core_str,
        logical_core.x,
        logical_core.y,
        virtual_core.x,
        virtual_core.y,
        risc,
        msg,
        kernel);

    if (assert_type == dev_msgs::DebugAssertTripped) {
        // Build regex pattern from string expected, replacing "on line 0" with "on line \d+"
        std::string pattern = regex_escape(expected);
        const std::string placeholder = "on line 0";
        size_t pos = pattern.find(placeholder);
        ASSERT_NE(pos, std::string::npos)
            << "Expected placeholder '" << placeholder << "' not found in escaped pattern: " << pattern;
        pattern.replace(pos, placeholder.length(), "on line \\d+");
        EXPECT_TRUE(std::regex_match(exception, std::regex(pattern)))
            << "Expected pattern: " << pattern << "\nActual: " << exception;
    } else {
        // Other assert types have fixed messages, exact match
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
        GTEST_SKIP() << "Test " << params.test_name << " requires processor type " << params.processor.processor_type
                     << " but only " << available_processors << " available on this architecture";
    }

    // Dispatch mode validation:
    // - IDLE_ETH cores only support SD (FD not yet implemented)
    // - TENSIX/ACTIVE_ETH cores: SD only used for Quasar watcher tests (TODO: Remove once FD enabled on Quasar)
    bool is_idle_eth = (params.processor.core_type == HalProgrammableCoreType::IDLE_ETH);
    bool is_quasar = (tt::tt_metal::MetalContext::instance().hal().get_arch() == tt::ARCH::QUASAR);
    bool using_slow_dispatch = this->IsSlowDispatch();

    if (is_idle_eth && !using_slow_dispatch) {
        log_info(tt::LogTest, "IDLE_ETH requires Slow Dispatch (Fast Dispatch not yet supported).");
        GTEST_SKIP();
    }
    if (using_slow_dispatch && !is_quasar && !is_idle_eth) {
        GTEST_SKIP() << "Slow Dispatch tests only run on Quasar or IDLE_ETH cores";
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
        // DM2 to DM7 only run on Quasar
        WatcherTestParams{"DM2", {TENSIX, DM, 2}, dev_msgs::DebugAssertTripped},
        WatcherTestParams{"DM3", {TENSIX, DM, 3}, dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped},
        WatcherTestParams{"DM4", {TENSIX, DM, 4}, dev_msgs::DebugAssertNCriscNOCNonpostedWritesSentTripped},
        WatcherTestParams{"DM5", {TENSIX, DM, 5}, dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped},
        WatcherTestParams{"DM6", {TENSIX, DM, 6}, dev_msgs::DebugAssertRtaOutOfBounds},
        WatcherTestParams{"DM7", {TENSIX, DM, 7}, dev_msgs::DebugAssertCrtaOutOfBounds},
        WatcherTestParams{"Trisc0", {TENSIX, COMPUTE, 0}, dev_msgs::DebugAssertNCriscNOCNonpostedWritesSentTripped},
        WatcherTestParams{"Trisc1", {TENSIX, COMPUTE, 1}, dev_msgs::DebugAssertNCriscNOCPostedWritesSentTripped},
        WatcherTestParams{"Trisc2", {TENSIX, COMPUTE, 2}, dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped},
        WatcherTestParams{
            "Trisc3", {TENSIX, COMPUTE, 3}, dev_msgs::DebugAssertRtaOutOfBounds},  // Trisc3 only Runs on Quasar
        WatcherTestParams{"Erisc", {ACTIVE_ETH, DM, 0}, dev_msgs::DebugAssertNCriscNOCNonpostedAtomicsFlushedTripped},
        WatcherTestParams{"IErisc", {IDLE_ETH, DM, 0}, dev_msgs::DebugAssertNCriscNOCReadsFlushedTripped}),
    [](const ::testing::TestParamInfo<WatcherTestParams>& info) { return info.param.test_name; });

}  // namespace
