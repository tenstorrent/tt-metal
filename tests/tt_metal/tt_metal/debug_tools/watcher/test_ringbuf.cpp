// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
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
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include <umd/device/types/core_coordinates.hpp>
#include "impl/debug/debug_helpers.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking debug ring buffer feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NUM_PUSHES_MULTI = 4;  // Multi-writer test: 4 pushes per thread (6 DMs, or 16 TRISCs)

// Expected strings for a single-processor ring buffer test.
// SPSC: pattern (idx << 16) | (idx + 1). MPSC: pattern (thread_idx << 16) | seq.
// Newest first, limited to buffer capacity (num_pushes is chosen by the caller to exceed
// capacity, so this always exercises wraparound).
std::vector<std::string> get_expected_single_processor(
    HalProgrammableCoreType core_type, uint32_t thread_idx, uint32_t num_pushes) {
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    bool is_mpsc = hal.has_mpsc_ring_buffer();
    uint32_t capacity = hal.get_ring_buffer_capacity();
    uint32_t first_visible = (num_pushes > capacity) ? num_pushes - capacity : 0;

    std::vector<uint32_t> data;
    for (uint32_t seq = num_pushes - 1; seq >= first_visible && seq < num_pushes; seq--) {
        data.push_back(is_mpsc ? ((thread_idx << 16) | seq) : ((seq << 16) | (seq + 1)));
    }

    std::vector<uint32_t> thread_indices;
    if (is_mpsc) {
        thread_indices.assign(data.size(), thread_idx);
    }
    std::vector<std::string> result = {"debug_ring_buffer="};
    auto lines = FormatRingBuffer(data, thread_indices, core_type);
    result.insert(result.end(), lines.begin(), lines.end());
    return result;
}

// Each thread's first push must be (hw_thread_id << 16) | 0, where hw_thread_id is the physical
// hw thread id (get_hw_thread_idx()) actually encoded by the kernel -- not the loop-local index.
// hw_id_offset maps the local index [0, num_threads) to the physical id range actually launched
// (e.g. Quasar user DMs launch on DM2..DM7, so hw_id_offset=2).
std::vector<std::string> get_expected_multi_writer(
    const std::function<std::string(uint32_t)>& prefix_for_thread, uint32_t num_threads, uint32_t hw_id_offset = 0) {
    std::vector<std::string> expected = {"debug_ring_buffer="};
    for (uint32_t local_idx = 0; local_idx < num_threads; local_idx++) {
        uint32_t hw_id = local_idx + hw_id_offset;
        expected.push_back(fmt::format("[{}]0x{:08x}", prefix_for_thread(hw_id), (hw_id << 16) | 0));
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
    Program program = Program();
    auto* device = mesh_device->get_devices()[0];
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    bool is_quasar = device->arch() == tt::ARCH::QUASAR;
    // Exceed capacity so every test exercises wraparound, regardless of buffer size.
    uint32_t num_pushes = multi_dm_test ? NUM_PUSHES_MULTI : (hal.get_ring_buffer_capacity() + 12);
    constexpr const char* kernel_legacy = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp";
    constexpr const char* kernel_metal2 = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf_2_0.cpp";
    const experimental::KernelSpecName kRingbufKernelName{"ringbuf_kernel"};

    // Depending on riscv type, choose one core to run the test on
    // and set up the kernel on the correct risc
    CoreCoord logical_core, virtual_core;
    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX: {
            logical_core = CoreCoord{0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            // TENSIX cores use the Metal 2.0 host API; ETH/DRAM/DISPATCH cores below remain on the
            // legacy host API (no Metal 2.0 equivalent exists for those programmable core types).
            experimental::KernelSpec kernel_spec{.unique_id = kRingbufKernelName, .source = kernel_metal2};
            switch (processor.processor_class) {
                case HalProcessorClassType::DM: {
                    kernel_spec.compile_time_args["num_pushes"] = num_pushes;
                    if (is_quasar) {
                        // No way to pin a Gen2 DM kernel to a specific hw thread: launch on all 6
                        // user DM threads (DM0/DM1 are reserved for the runtime) and filter in-kernel.
                        kernel_spec.num_threads = 6;
                        if (multi_dm_test) {
                            kernel_spec.compiler_options.defines = {{"MULTI_DM_TEST", "1"}};
                        } else {
                            kernel_spec.compile_time_args["dm_id"] = processor.processor_type;
                        }
                        kernel_spec.hw_config = experimental::DataMovementGen2Config{};
                    } else {
                        kernel_spec.hw_config = experimental::DataMovementGen1Config{
                            .processor = static_cast<tt_metal::DataMovementProcessor>(processor.processor_type),
                            .noc = (processor.processor_type == 0) ? tt_metal::NOC::RISCV_0_default
                                                                   : tt_metal::NOC::RISCV_1_default,
                        };
                    }
                    break;
                }
                case HalProcessorClassType::COMPUTE: {
                    kernel_spec.compile_time_args["num_pushes"] = num_pushes;
                    if (multi_dm_test) {
                        kernel_spec.compiler_options.defines = {{"MULTI_DM_TEST", "1"}};
                    } else {
                        kernel_spec.compiler_options.defines = {
                            {fmt::format("WATCHER_RINGBUF_TRISC{}", processor.processor_type), "1"}};
                    }
                    if (is_quasar) {
                        kernel_spec.num_threads = multi_dm_test ? 4 : 1;
                        kernel_spec.hw_config = experimental::ComputeGen2Config{};
                    } else {
                        kernel_spec.hw_config = experimental::ComputeGen1Config{};
                    }
                    break;
                }
            }
            experimental::WorkUnitSpec wu{
                .name = "main",
                .kernels = {kRingbufKernelName},
                .target_nodes = experimental::NodeCoord{logical_core},
            };
            experimental::ProgramSpec spec{.name = "watcher_ringbuf", .kernels = {kernel_spec}, .work_units = {wu}};
            program = experimental::MakeProgramFromSpec(*mesh_device, spec);
            break;
        }
        case HalProgrammableCoreType::ACTIVE_ETH:
            if (device->get_active_ethernet_cores(true).empty()) {
                log_info(LogTest, "Skipping this test since device has no active ethernet cores.");
                GTEST_SKIP();
            }
            logical_core = *(device->get_active_ethernet_cores(true).begin());
            virtual_core = device->ethernet_core_from_logical_core(logical_core);
            CreateKernel(
                program,
                kernel_legacy,
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
                kernel_legacy,
                logical_core,
                EthernetConfig{.eth_mode = Eth::IDLE, .noc = tt_metal::NOC::NOC_0, .compile_args = {num_pushes}});
            break;
        case HalProgrammableCoreType::DRAM: {
            if (!hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
                log_info(LogTest, "Skipping: DRAM programmable cores not available on this architecture.");
                GTEST_SKIP();
            }
            // Subchannel 0 is the syseng-owned NOC0 DRAM endpoint (no DRISC firmware); use subchannel 1.
            logical_core = CoreCoord{0, 1};
            virtual_core = device->virtual_core_from_logical_core(logical_core, CoreType::DRAM);
            CreateKernel(
                program,
                kernel_legacy,
                logical_core,
                DramConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {num_pushes}});
            break;
        }
        case HalProgrammableCoreType::DISPATCH: {
            if (!hal.has_programmable_core_type(HalProgrammableCoreType::DISPATCH)) {
                log_info(LogTest, "Skipping: dispatch-engine programmable cores not available on this architecture.");
                GTEST_SKIP();
            }
            log_info(LogTest, "Skipping: watcher ringbuf test not yet supported on dispatch-engine cores.");
            GTEST_SKIP();
        }
        case HalProgrammableCoreType::COUNT: TT_THROW("Unsupported core type");
    }
    log_info(LogTest, "Running test on device {} core {}[{}]...", device->id(), logical_core, virtual_core);
    workload.add_program(device_range, std::move(program));

    // Run the program
    fixture->RunProgram(mesh_device, workload, true);

    log_info(tt::LogTest, "Checking file: {}", fixture->log_file_name);

    // Check log
    if (multi_dm_test) {
        if (processor.processor_class == HalProcessorClassType::DM) {
            // DM0/DM1 are reserved for the runtime and never assigned to a kernel (see
            // ReserveProcessors in program_spec.cpp, which fills bits in ascending order and
            // treats DM0/DM1 as pre-used), so the 6 launched threads land on DM2..DM7 in order.
            EXPECT_TRUE(FileContainsAllStrings(
                fixture->log_file_name,
                get_expected_multi_writer(
                    [](uint32_t dm) { return fmt::format("DM{}", dm); }, /*num_threads=*/6, /*hw_id_offset=*/2)));
        } else {
            // All 16 TRISCs (4 engines x 4 roles) push; HAL processor index for COMPUTE starts
            // right after the 8 DM entries.
            EXPECT_TRUE(FileContainsAllStrings(
                fixture->log_file_name,
                get_expected_multi_writer(
                    [&hal](uint32_t hw_id) {
                        return hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, hw_id, false);
                    },
                    /*num_threads=*/16,
                    /*hw_id_offset=*/8)));
        }
    } else {
        // Thread index for DM is processor_type (0-7), for TRISC it's 8+ based on HAL mapping
        uint32_t thread_idx = processor.processor_type;
        if (processor.processor_class == HalProcessorClassType::COMPUTE) {
            // Compute processors start after DM processors in the HAL index
            thread_idx =
                hal.get_processor_index(processor.core_type, processor.processor_class, processor.processor_type);
        }
        EXPECT_TRUE(FileContainsAllStringsInOrder(
            fixture->log_file_name, get_expected_single_processor(processor.core_type, thread_idx, num_pushes)));
    }
}

void RunMultiRiscTestBH(MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    CoreCoord logical_core{0, 0};
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf_2_0.cpp";

    const experimental::KernelSpecName brisc_name{"brisc"};
    const experimental::KernelSpecName ncrisc_name{"ncrisc"};
    const experimental::KernelSpecName trisc_name{"trisc"};

    experimental::KernelSpec brisc_spec{
        .unique_id = brisc_name,
        .source = kernel_path,
        .compile_time_args = {{"num_pushes", NUM_PUSHES_MULTI}},
        .hw_config =
            experimental::DataMovementGen1Config{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default},
    };
    experimental::KernelSpec ncrisc_spec{
        .unique_id = ncrisc_name,
        .source = kernel_path,
        .compile_time_args = {{"num_pushes", NUM_PUSHES_MULTI}},
        .hw_config =
            experimental::DataMovementGen1Config{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default},
    };
    // One ComputeGen1Config kernel builds all 3 TRISC binaries; each needs its own
    // WATCHER_RINGBUF_TRISC{n} define.
    experimental::KernelSpec trisc_spec{
        .unique_id = trisc_name,
        .source = kernel_path,
        .compiler_options =
            {.defines =
                 {{"WATCHER_RINGBUF_TRISC0", "1"}, {"WATCHER_RINGBUF_TRISC1", "1"}, {"WATCHER_RINGBUF_TRISC2", "1"}}},
        .compile_time_args = {{"num_pushes", NUM_PUSHES_MULTI}},
        .hw_config = experimental::ComputeGen1Config{},
    };

    experimental::WorkUnitSpec wu{
        .name = "main",
        .kernels = {brisc_name, ncrisc_name, trisc_name},
        .target_nodes = experimental::NodeCoord{logical_core},
    };
    experimental::ProgramSpec spec{
        .name = "watcher_ringbuf_multi_risc",
        .kernels = {brisc_spec, ncrisc_spec, trisc_spec},
        .work_units = {wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);
    workload.add_program(device_range, std::move(program));

    fixture->RunProgram(mesh_device, workload, true);

    log_info(tt::LogTest, "Checking file: {}", fixture->log_file_name);
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    EXPECT_TRUE(FileContainsAllStrings(
        fixture->log_file_name,
        get_expected_multi_writer(
            [&hal](uint32_t thread_idx) {
                return hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, thread_idx, false);
            },
            /*num_threads=*/5)));
}

// Test parameters for the single-processor (and multi-writer) ring buffer suite.
struct RingBufferTestParams {
    std::string test_name;
    HalProcessorIdentifier processor;
    bool multi_dm_test = false;
};

class WatcherRingBufferTest : public MeshWatcherFixture, public ::testing::WithParamInterface<RingBufferTestParams> {};

TEST_P(WatcherRingBufferTest, TestWatcherRingBuffer) {
    const auto& params = GetParam();
    const auto& hal = MetalContext::instance().hal();
    bool is_quasar = (hal.get_arch() == tt::ARCH::QUASAR);

    if (!hal.has_programmable_core_type(params.processor.core_type)) {
        GTEST_SKIP() << "Test " << params.test_name << ": core type not available on this architecture";
    }

    uint32_t available_processors = hal.get_processor_types_count(
        params.processor.core_type, static_cast<uint32_t>(params.processor.processor_class));
    if (params.processor.processor_type >= available_processors) {
        GTEST_SKIP() << "Test " << params.test_name << " requires processor type " << params.processor.processor_type
                     << " but only " << available_processors << " available on this architecture";
    }

    // Multi-writer test is Quasar-only (uses the Quasar-specific 6-DM/16-TRISC reservation scheme).
    if (params.multi_dm_test && !is_quasar) {
        GTEST_SKIP() << "Multi-writer MPSC test is Quasar-only";
    }

    // On Quasar, DM0/DM1 are reserved for the runtime (ISR/remapper) and never assigned to a
    // user kernel, so the single-DM Brisc/NCrisc params (processor_type 0/1, meaningful only as
    // BRISC/NCRISC on Gen1 WH/BH) can't be exercised there.
    bool is_reserved_quasar_dm =
        is_quasar && !params.multi_dm_test && params.processor.processor_class == HalProcessorClassType::DM &&
        params.processor.core_type == HalProgrammableCoreType::TENSIX && params.processor.processor_type < 2;
    if (is_reserved_quasar_dm) {
        GTEST_SKIP() << "Test " << params.test_name << ": DM0/DM1 are reserved for the runtime on Quasar";
    }

    bool is_idle_eth = (params.processor.core_type == HalProgrammableCoreType::IDLE_ETH);
    bool is_dram = (params.processor.core_type == HalProgrammableCoreType::DRAM);
    if ((is_idle_eth || is_dram) && !this->IsSlowDispatch()) {
        GTEST_SKIP() << "Test " << params.test_name << " requires Slow Dispatch";
    }

    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, params.processor, params.multi_dm_test);
                if (params.multi_dm_test) {
                    RunTest(
                        fixture,
                        mesh_device,
                        {HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0},
                        /*multi_dm_test=*/true);
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
        // Ethernet processors
        RingBufferTestParams{"Erisc", {ACTIVE_ETH, DM, 0}},
        RingBufferTestParams{"IErisc", {IDLE_ETH, DM, 0}},
        // DRAM (DRISC)
        RingBufferTestParams{"Drisc", {DRAM, DM, 0}},
        // Multi-writer MPSC test (Quasar only): 6 user DMs, then 16 TRISCs
        RingBufferTestParams{"MpscMultiDM", {TENSIX, DM, 0}, /*multi_dm_test=*/true}),
    [](const ::testing::TestParamInfo<RingBufferTestParams>& info) { return info.param.test_name; });

TEST_F(MeshWatcherFixture, TestWatcherRingBufferMpscMultiRiscBH) {
    for (auto& mesh_device : this->devices_) {
        auto* device = mesh_device->get_devices()[0];
        if (device->arch() != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Multi-RISC MPSC test is Blackhole-only";
        }
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunMultiRiscTestBH(fixture, mesh_device);
            },
            mesh_device);
    }
}

}  // namespace
