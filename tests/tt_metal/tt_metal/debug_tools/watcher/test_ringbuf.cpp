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
#include <tt-metalium/kernel_types.hpp>
#include <tt_stl/assert.hpp>
#include "debug_tools_fixture.hpp"
#include "debug_tools_test_utils.hpp"
#include "hal_types.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include "impl/kernels/kernel.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/debug/debug_helpers.hpp"
#include "hostdev/debug_ring_buffer_common.h"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking debug ring buffer feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

namespace {
const std::string kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp";

// Pushes per processor for multi-threaded test (24 x 5 = 120, fits in 128-element MPSC buffer)
constexpr uint32_t PUSHES_PER_PROCESSOR_MULTI = 5;

// Expected strings for single-processor tests
// Pattern: SPSC uses (idx << 16) | (idx + 1), MPSC uses (thread_idx << 16) | seq
// Newest first, limited to buffer capacity
std::vector<std::string> get_expected_single_processor(
    HalProgrammableCoreType core_type, uint32_t thread_idx, uint32_t num_pushes) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    bool is_mpsc = hal.has_mpsc_ring_buffer();
    uint32_t capacity = hal.get_ring_buffer_capacity();
    uint32_t first_visible = (num_pushes > capacity) ? num_pushes - capacity : 0;

    std::vector<std::string> result = {"debug_ring_buffer="};
    for (uint32_t seq = num_pushes - 1; seq >= first_visible && seq < num_pushes; seq--) {
        if (is_mpsc) {
            auto proc_name = hal.get_processor_class_name(core_type, thread_idx, false);
            result.push_back(fmt::format("[{}]0x{:08x}", proc_name, (thread_idx << 16) | seq));
        } else {
            result.push_back(fmt::format("0x{:08x}", (seq << 16) | (seq + 1)));
        }
    }
    return result;
}

// Expected strings for Quasar multi-threaded test (all DMs + TRISCs)
// Verifies all processors wrote all entries (order is non-deterministic)
std::vector<std::string> get_expected_multi_threaded(
    HalProgrammableCoreType core_type, uint32_t num_processors, uint32_t pushes_per_proc) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();

    std::vector<std::string> expected = {"debug_ring_buffer="};
    for (uint32_t proc_idx = 0; proc_idx < num_processors; proc_idx++) {
        auto proc_name = hal.get_processor_class_name(core_type, proc_idx, false);
        for (uint32_t seq = 0; seq < pushes_per_proc; seq++) {
            expected.push_back(fmt::format("[{}]0x{:08x}", proc_name, (proc_idx << 16) | seq));
        }
    }
    return expected;
}

void RunMultiThreadedTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProgrammableCoreType core_type) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    uint32_t total_processors = hal.get_max_processors_per_core();

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, {});
    auto& program = workload.get_programs().at(device_range);
    CoreCoord logical_core = {0, 0};

    tt::tt_metal::experimental::quasar::CreateKernel(
        program,
        kernel,
        logical_core,
        tt::tt_metal::experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 8,
            .compile_args = {PUSHES_PER_PROCESSOR_MULTI},
            .defines = {{"MULTI_THREADED_TEST", "1"}}});

    tt::tt_metal::experimental::quasar::CreateKernel(
        program,
        kernel,
        logical_core,
        tt::tt_metal::experimental::quasar::QuasarComputeConfig{
            .num_threads_per_cluster = 4,
            .compile_args = {PUSHES_PER_PROCESSOR_MULTI},
            .defines = {{"MULTI_THREADED_TEST", "1"}}});

    log_info(LogTest, "Running multi-threaded test on all processors)...");
    fixture->RunProgram(mesh_device, workload, true);

    auto expected = get_expected_multi_threaded(core_type, total_processors, PUSHES_PER_PROCESSOR_MULTI);
    EXPECT_TRUE(FileContainsAllStrings(fixture->log_file_name, expected));
}

void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProcessorIdentifier processor) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    bool is_quasar = hal.get_arch() == tt::ARCH::QUASAR;

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, {});
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];
    CoreCoord logical_core = {0, 0};
    CoreCoord virtual_core = device->worker_core_from_logical_core(logical_core);

    uint32_t num_pushes = hal.get_ring_buffer_capacity() + 12;  // overflow to test wraparound

    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX:
            switch (processor.processor_class) {
                case HalProcessorClassType::DM: {
                    if (is_quasar) {
                        tt::tt_metal::experimental::quasar::CreateKernel(
                            program,
                            kernel,
                            logical_core,
                            tt::tt_metal::experimental::quasar::QuasarDataMovementConfig{
                                .num_threads_per_cluster = 8, .compile_args = {num_pushes, processor.processor_type}});
                    } else {
                        DataMovementConfig dm_config{
                            .processor = static_cast<tt_metal::DataMovementProcessor>(processor.processor_type),
                            .noc = (processor.processor_type == 0) ? tt_metal::NOC::RISCV_0_default
                                                                   : tt_metal::NOC::RISCV_1_default,
                            .compile_args = {num_pushes}};
                        CreateKernel(program, kernel, logical_core, dm_config);
                    }
                    break;
                }
                case HalProcessorClassType::COMPUTE:
                    if (is_quasar) {
                        tt::tt_metal::experimental::quasar::CreateKernel(
                            program,
                            kernel,
                            logical_core,
                            tt::tt_metal::experimental::quasar::QuasarComputeConfig{
                                .num_threads_per_cluster = 1,
                                .compile_args = {num_pushes},
                                .defines = {{fmt::format("TRISC{}", processor.processor_type), "1"}}});
                    } else {
                        CreateKernel(
                            program,
                            kernel,
                            logical_core,
                            ComputeConfig{
                                .compile_args = {num_pushes},
                                .defines = {{fmt::format("TRISC{}", processor.processor_type), "1"}}});
                    }
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
                kernel,
                logical_core,
                EthernetConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {num_pushes}});
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
                kernel,
                logical_core,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .compile_args = {num_pushes}});
            break;
        case HalProgrammableCoreType::DRAM:
            log_info(LogTest, "Skipping: DRAM cores do not support watcher ring buffer tests.");
            GTEST_SKIP();
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported core type");
    }
    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, virtual_core);

    // Run the program
    fixture->RunProgram(mesh_device, workload, true);

    log_info(tt::LogTest, "Checking file: {}", fixture->log_file_name);

    // Validate output
    uint32_t thread_idx = processor.processor_type;
    if (is_quasar && processor.processor_class == HalProcessorClassType::COMPUTE) {
        thread_idx = hal.get_processor_index(processor.core_type, processor.processor_class, processor.processor_type);
    }
    EXPECT_TRUE(FileContainsAllStringsInOrder(
        fixture->log_file_name, get_expected_single_processor(processor.core_type, thread_idx, num_pushes)));
}

// Test parameters
struct RingBufferTestParams {
    std::string test_name;
    HalProcessorIdentifier processor;
    bool multi_threaded = false;
};

class WatcherRingBufferTest : public MeshWatcherFixture, public ::testing::WithParamInterface<RingBufferTestParams> {};

TEST_P(WatcherRingBufferTest, TestWatcherRingBuffer) {
    const auto& params = GetParam();
    const auto& hal = MetalContext::instance().hal();

    // Skip if processor type not available on this architecture
    uint32_t core_type_index = hal.get_programmable_core_type_index(params.processor.core_type);
    uint32_t available_processors =
        hal.get_processor_types_count(core_type_index, static_cast<uint32_t>(params.processor.processor_class));

    if (params.processor.processor_type >= available_processors) {
        GTEST_SKIP() << "Test " << params.test_name << " requires processor type " << params.processor.processor_type
                     << " but only " << available_processors << " available on this architecture";
    }

    // Multi-threaded test is Quasar-only
    if (params.multi_threaded && hal.get_arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Multi-threaded test is Quasar-only";
    }

    // Quasar and IDLE_ETH require slow dispatch
    bool is_quasar = (hal.get_arch() == tt::ARCH::QUASAR);
    bool is_idle_eth = (params.processor.core_type == HalProgrammableCoreType::IDLE_ETH);
    if ((is_quasar || is_idle_eth) && !this->IsSlowDispatch()) {
        GTEST_SKIP() << "Quasar and IDLE_ETH require Slow Dispatch";
    }

    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                if (params.multi_threaded) {
                    RunMultiThreadedTest(fixture, mesh_device, params.processor.core_type);
                } else {
                    RunTest(fixture, mesh_device, params.processor);
                }
            },
            mesh_device);
    }
}

using enum HalProgrammableCoreType;
using enum HalProcessorClassType;

INSTANTIATE_TEST_SUITE_P(
    WatcherRingBufferTests,
    WatcherRingBufferTest,
    ::testing::Values(
        // DM processors
        RingBufferTestParams{"Brisc", {TENSIX, DM, 0}},
        RingBufferTestParams{"NCrisc", {TENSIX, DM, 1}},
        RingBufferTestParams{"DM2", {TENSIX, DM, 2}},
        RingBufferTestParams{"DM3", {TENSIX, DM, 3}},
        RingBufferTestParams{"DM4", {TENSIX, DM, 4}},
        RingBufferTestParams{"DM5", {TENSIX, DM, 5}},
        RingBufferTestParams{"DM6", {TENSIX, DM, 6}},
        RingBufferTestParams{"DM7", {TENSIX, DM, 7}},
        // TRISC processors
        RingBufferTestParams{"Trisc0", {TENSIX, COMPUTE, 0}},
        RingBufferTestParams{"Trisc1", {TENSIX, COMPUTE, 1}},
        RingBufferTestParams{"Trisc2", {TENSIX, COMPUTE, 2}},
        RingBufferTestParams{"Trisc3", {TENSIX, COMPUTE, 3}},  // Quasar only
        // Ethernet processors
        RingBufferTestParams{"Erisc", {ACTIVE_ETH, DM, 0}},
        RingBufferTestParams{"IErisc", {IDLE_ETH, DM, 0}},
        // Multi-threaded test (Quasar only, all 24 processors)
        RingBufferTestParams{"MultiThreaded", {TENSIX, DM, 0}, /*multi_threaded=*/true}),
    [](const ::testing::TestParamInfo<RingBufferTestParams>& info) { return info.param.test_name; });

}  // namespace
