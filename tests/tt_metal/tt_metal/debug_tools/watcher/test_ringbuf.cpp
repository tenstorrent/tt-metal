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

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking debug ring buffer feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NUM_PUSHES_SINGLE = 40;  // Single-thread tests push 40 values
constexpr uint32_t NUM_PUSHES_MULTI = 4;    // Multi-DM test: 4 pushes per DM (8 DMs x 4 = 32 total)

// Generate expected ring buffer data for TRISC/ERISC/BH/WH tests
// Pattern: (idx << 16) | (idx + 1), buffer holds 32, newest first (40 down to 9)
std::vector<uint32_t> get_expected_data() {
    std::vector<uint32_t> data;
    for (uint32_t idx = NUM_PUSHES_SINGLE - 1; idx >= NUM_PUSHES_SINGLE - 32; idx--) {
        data.push_back((idx << 16) | (idx + 1));
    }
    return data;
}

// Generate expected ring buffer data for Quasar single-DM tests
// Pattern: (thread_idx << 16) | seq, buffer holds 32, newest first
std::vector<uint32_t> get_expected_data_dm(uint32_t thread_idx) {
    std::vector<uint32_t> data;
    for (uint32_t seq = NUM_PUSHES_SINGLE - 1; seq >= NUM_PUSHES_SINGLE - 32; seq--) {
        data.push_back((thread_idx << 16) | seq);
    }
    return data;
}

// Expected strings for BH/WH (SPSC format)
std::vector<std::string> get_expected_spsc() {
    auto data = get_expected_data();
    std::vector<std::string> result = {"debug_ring_buffer="};
    auto lines = FormatRingBuffer(data);
    result.insert(result.end(), lines.begin(), lines.end());
    return result;
}

// Expected strings for Quasar single-DM (MPSC format with processor prefix)
std::vector<std::string> get_expected_mpsc(HalProgrammableCoreType core_type, uint32_t thread_idx) {
    auto data = get_expected_data_dm(thread_idx);
    std::vector<uint32_t> thread_indices(data.size(), thread_idx);  // All from same thread in test
    std::vector<std::string> result = {"debug_ring_buffer="};
    auto lines = FormatRingBuffer(data, thread_indices, core_type);
    result.insert(result.end(), lines.begin(), lines.end());
    return result;
}

// Expected strings for Quasar multi-DM test (MPSC format)
// Verifies all 8 DMs wrote entries with matching [DMx] prefix and data
std::vector<std::string> get_expected_multi_dm() {
    std::vector<std::string> expected = {"debug_ring_buffer="};
    for (uint32_t dm = 0; dm < 8; dm++) {
        // First push from each DM: (dm << 16) | 0
        expected.push_back(fmt::format("[DM{}]0x{:08x}", dm, (dm << 16) | 0));
    }
    return expected;
}

namespace {

void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProcessorIdentifier processor,
    bool multi_dm_test = false) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, {});
    auto& program = workload.get_programs().at(device_range);
    auto* device = mesh_device->get_devices()[0];
    bool is_quasar = device->arch() == tt::ARCH::QUASAR;

    // Depending on riscv type, choose one core to run the test on
    // and set up the kernel on the correct risc
    CoreCoord logical_core, virtual_core;
    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX:
            logical_core = CoreCoord{0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            switch (processor.processor_class) {
                case HalProcessorClassType::DM: {
                    if (is_quasar) {
                        uint32_t num_pushes = multi_dm_test ? NUM_PUSHES_MULTI : NUM_PUSHES_SINGLE;
                        std::vector<uint32_t> compile_args =
                            multi_dm_test ? std::vector<uint32_t>{num_pushes}
                                          : std::vector<uint32_t>{num_pushes, processor.processor_type};
                        std::map<std::string, std::string> defines =
                            multi_dm_test ? std::map<std::string, std::string>{{"MULTI_DM_TEST", "1"}}
                                          : std::map<std::string, std::string>{};
                        tt::tt_metal::experimental::quasar::CreateKernel(
                            program,
                            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                            logical_core,
                            tt::tt_metal::experimental::quasar::QuasarDataMovementConfig{
                                .num_threads_per_cluster = 8, .compile_args = compile_args, .defines = defines});
                    } else {
                        DataMovementConfig dm_config{};
                        dm_config.processor = static_cast<tt_metal::DataMovementProcessor>(processor.processor_type);
                        dm_config.noc = (processor.processor_type == 0) ? tt_metal::NOC::RISCV_0_default
                                                                        : tt_metal::NOC::RISCV_1_default;
                        dm_config.compile_args = {NUM_PUSHES_SINGLE};
                        CreateKernel(
                            program,
                            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                            logical_core,
                            dm_config);
                    }
                    break;
                }
                case HalProcessorClassType::COMPUTE:
                    if (is_quasar) {
                        uint32_t num_threads = multi_dm_test ? 4 : 1;
                        uint32_t num_pushes = multi_dm_test ? NUM_PUSHES_MULTI : NUM_PUSHES_SINGLE;
                        std::map<std::string, std::string> defines =
                            multi_dm_test ? std::map<std::string, std::string>{{"MULTI_DM_TEST", "1"}}
                                          : std::map<std::string, std::string>{
                                                {fmt::format("TRISC{}", processor.processor_type), "1"}};
                        tt::tt_metal::experimental::quasar::CreateKernel(
                            program,
                            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                            logical_core,
                            experimental::quasar::QuasarComputeConfig{
                                .num_threads_per_cluster = num_threads,
                                .compile_args = {num_pushes},
                                .defines = defines});
                    } else {
                        CreateKernel(
                            program,
                            "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                            logical_core,
                            ComputeConfig{
                                .compile_args = {NUM_PUSHES_SINGLE},
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
                "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp",
                logical_core,
                EthernetConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {NUM_PUSHES_SINGLE}});
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
                EthernetConfig{
                    .eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .compile_args = {NUM_PUSHES_SINGLE}});
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

    // Check log
    if (is_quasar) {
        if (multi_dm_test) {
            EXPECT_TRUE(FileContainsAllStrings(fixture->log_file_name, get_expected_multi_dm()));
        } else {
            // Thread index for DM is processor_type (0-7), for TRISC it's 8+ based on HAL mapping
            uint32_t thread_idx = processor.processor_type;
            if (processor.processor_class == HalProcessorClassType::COMPUTE) {
                // Compute processors start after DM processors in the HAL index
                const auto& hal = tt::tt_metal::MetalContext::instance().hal();
                thread_idx =
                    hal.get_processor_index(processor.core_type, processor.processor_class, processor.processor_type);
            }
            EXPECT_TRUE(FileContainsAllStringsInOrder(
                fixture->log_file_name, get_expected_mpsc(processor.core_type, thread_idx)));
        }
    } else {
        EXPECT_TRUE(FileContainsAllStringsInOrder(fixture->log_file_name, get_expected_spsc()));
    }
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

TEST_F(MeshWatcherFixture, TestWatcherRingBufferMpscMultiDM) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        if (device->arch() != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "Multi-DM MPSC test is Quasar-only";
        }
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, {TENSIX, DM, 0}, /*multi_dm_test=*/true);
            },
            mesh_device);
    }
}

}  // namespace
