// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <functional>
#include <optional>
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
#include <tt-umd/types/core_coordinates.hpp>
#include "impl/debug/debug_helpers.hpp"
#include "impl/metal2_host_api/helpers.hpp"
#include "distributed/mesh_device_impl.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// A test for checking debug ring buffer feature.
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;
using namespace tt::tt_metal;

// 22 Quasar writers x 5 = 110 entries, within the 128-entry buffer so nothing is evicted
constexpr uint32_t NUM_PUSHES_MULTI = 5;

// Newest-first, limited to buffer capacity.
std::vector<std::string> get_expected_single_processor(
    const Hal& hal, HalProgrammableCoreType core_type, uint32_t thread_idx, uint32_t num_pushes) {
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
    auto lines = FormatRingBuffer(hal, data, thread_indices, core_type);
    result.insert(result.end(), lines.begin(), lines.end());
    return result;
}

// The kernel encodes get_hw_thread_idx(), so hw_id_offset shifts the loop index onto the physical
// ids actually launched (Quasar user DMs run on DM2...DM7, so hw_id_offset=2)
void append_expected_writers(
    std::vector<std::string>& expected,
    const std::function<std::string(uint32_t)>& prefix_for_thread,
    uint32_t num_threads,
    uint32_t hw_id_offset = 0) {
    for (uint32_t local_idx = 0; local_idx < num_threads; local_idx++) {
        uint32_t hw_id = local_idx + hw_id_offset;
        for (uint32_t seq = 0; seq < NUM_PUSHES_MULTI; seq++) {
            expected.push_back(fmt::format("[{}]0x{:08x}", prefix_for_thread(hw_id), (hw_id << 16) | seq));
        }
    }
}

namespace {

void RunTest(
    MeshWatcherFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    HalProcessorIdentifier processor) {
    // Set up program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram(*mesh_device);
    auto* device = mesh_device->impl().get_devices()[0];

    // Depending on riscv type, choose one core to run the test on
    // and set up the kernel on the correct risc
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    bool is_quasar = device->arch() == tt::ARCH::QUASAR;
    // Push past capacity so the oldest entries are overwritten, whatever the buffer size.
    uint32_t num_pushes = hal.get_ring_buffer_capacity() + 12;
    constexpr const char* kernel_legacy = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf.cpp";
    constexpr const char* kernel_metal2 = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf_2_0.cpp";
    const experimental::KernelSpecName kRingbufKernelName{"ringbuf_kernel"};

    CoreCoord logical_core, virtual_core;
    switch (processor.core_type) {
        case HalProgrammableCoreType::TENSIX: {
            logical_core = CoreCoord{0, 0};
            virtual_core = device->worker_core_from_logical_core(logical_core);
            // ETH/DRAM below stay on the legacy host API; it has no Metal 2.0 equivalent.
            experimental::KernelSpec kernel_spec{.unique_id = kRingbufKernelName, .source = kernel_metal2};
            switch (processor.processor_class) {
                case HalProcessorClassType::DM: {
                    kernel_spec.compile_time_args["num_pushes"] = num_pushes;
                    if (is_quasar) {
                        // Launch on all 6 user DM threads (DM0/DM1 are reserved for the runtime) and filter in-kernel.
                        kernel_spec.num_threads = 6;
                        kernel_spec.compile_time_args["dm_id"] = processor.processor_type;
                        kernel_spec.hw_config = experimental::DataMovementHardwareConfig{};
                    } else {
                        kernel_spec.hw_config = experimental::DataMovementHardwareConfig{
                            .config_1xx =
                                experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                                    .processor = static_cast<tt_metal::DataMovementProcessor>(processor.processor_type),
                                    .noc = (processor.processor_type == 0) ? tt_metal::NOC::RISCV_0_default
                                                                           : tt_metal::NOC::RISCV_1_default,
                                },
                        };
                    }
                    break;
                }
                case HalProcessorClassType::COMPUTE: {
                    kernel_spec.compile_time_args["num_pushes"] = num_pushes;
                    kernel_spec.compiler_options.defines = {
                        {fmt::format("WATCHER_RINGBUF_TRISC{}", processor.processor_type), "1"}};
                    if (is_quasar) {
                        kernel_spec.num_threads = 1;
                        kernel_spec.hw_config = experimental::ComputeHardwareConfig{};
                    } else {
                        kernel_spec.hw_config = experimental::ComputeHardwareConfig{};
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
    uint32_t thread_idx =
        hal.get_processor_index(processor.core_type, processor.processor_class, processor.processor_type);
    EXPECT_TRUE(FileContainsAllStringsInOrder(
        fixture->log_file_name, get_expected_single_processor(hal, processor.core_type, thread_idx, num_pushes)));
}

// sem_wrap (Quasar only): the kernel starts the head so that half the pushes land before the rewind, then
// every entry and the final head are checked.
void RunMultiWriterTest(
    MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool sem_wrap) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    const auto& hal = tt::tt_metal::MetalContext::instance().hal();
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const bool is_quasar = mesh_device->arch() == tt::ARCH::QUASAR;
    CoreCoord logical_core{0, 0};
    const tt_cxy_pair core{
        static_cast<size_t>(mesh_device->impl().get_devices()[0]->id()),
        mesh_device->worker_core_from_logical_core(logical_core)};
    std::optional<uint32_t> expected_head;
    // Where the kernel hands the final head to the host.
    const auto head_addr =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED));
    constexpr const char* kernel = "tests/tt_metal/tt_metal/test_kernels/misc/watcher_ringbuf_2_0.cpp";

    std::vector<experimental::KernelSpec> specs;
    std::vector<experimental::KernelSpecName> names;
    auto add_spec = [&](const char* name, experimental::KernelSpec spec) {
        spec.unique_id = experimental::KernelSpecName{name};
        spec.source = kernel;
        spec.compile_time_args["num_pushes"] = NUM_PUSHES_MULTI;
        if (sem_wrap) {
            spec.compiler_options.defines["WATCHER_RINGBUF_SEM_WRAP"] = "1";
            spec.compile_time_args["num_dm_threads"] = experimental::QUASAR_USER_DM_CORES_PER_NODE;
            spec.compile_time_args["num_tensixes"] = experimental::QUASAR_TENSIX_ENGINES_PER_NODE;
            spec.compile_time_args["head_addr"] = head_addr;
        }
        names.push_back(spec.unique_id);
        specs.push_back(std::move(spec));
    };
    auto tensix_name = [&hal](uint32_t hw_id) {
        return hal.get_processor_class_name(HalProgrammableCoreType::TENSIX, hw_id, false);
    };

    std::vector<std::string> expected = {"debug_ring_buffer="};
    if (is_quasar) {
        add_spec(
            "dm",
            {.num_threads = experimental::QUASAR_USER_DM_CORES_PER_NODE,
             .compiler_options = {.defines = {{"MULTI_DM_TEST", "1"}}},
             .hw_config = experimental::DataMovementHardwareConfig{}});
        add_spec(
            "compute",
            {.num_threads = experimental::QUASAR_TENSIX_ENGINES_PER_NODE,
             .compiler_options = {.defines = {{"MULTI_DM_TEST", "1"}}},
             .hw_config = experimental::ComputeHardwareConfig{}});
        // DM0/DM1 are reserved, so the launched threads land on DM2...DM7.
        const uint32_t num_trisc_writers = hal.get_processor_types_count(
            HalProgrammableCoreType::TENSIX, static_cast<uint32_t>(HalProcessorClassType::COMPUTE));
        const uint32_t first_trisc_hw_id =
            hal.get_processor_index(HalProgrammableCoreType::TENSIX, HalProcessorClassType::COMPUTE, 0);
        append_expected_writers(
            expected,
            tensix_name,
            experimental::QUASAR_USER_DM_CORES_PER_NODE,
            experimental::QUASAR_RESERVED_DM_CORES_PER_NODE);
        append_expected_writers(expected, tensix_name, num_trisc_writers, first_trisc_hw_id);
        if (sem_wrap) {
            const uint32_t total_pushes =
                (experimental::QUASAR_USER_DM_CORES_PER_NODE + num_trisc_writers) * NUM_PUSHES_MULTI;
            expected_head = total_pushes - total_pushes / 2;
            // Cleared so a kernel that never hands the head over fails the check.
            const uint32_t zero = 0;
            cluster.write_core(&zero, sizeof(zero), core, head_addr);
        }
    } else {
        add_spec(
            "brisc",
            {.hw_config = experimental::DataMovementHardwareConfig{
                 .config_1xx =
                     experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                         .processor = tt_metal::DataMovementProcessor::RISCV_0,
                         .noc = tt_metal::NOC::RISCV_0_default,
                     },
             }});
        add_spec(
            "ncrisc",
            {.hw_config = experimental::DataMovementHardwareConfig{
                 .config_1xx =
                     experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                         .processor = tt_metal::DataMovementProcessor::RISCV_1,
                         .noc = tt_metal::NOC::RISCV_1_default,
                     },
             }});
        // One compute kernel builds all 3 TRISC binaries; each needs its own define.
        add_spec(
            "trisc",
            {.compiler_options =
                 {.defines =
                      {{"WATCHER_RINGBUF_TRISC0", "1"},
                       {"WATCHER_RINGBUF_TRISC1", "1"},
                       {"WATCHER_RINGBUF_TRISC2", "1"}}},
             .hw_config = experimental::ComputeHardwareConfig{}});
        append_expected_writers(expected, tensix_name, 5);
    }

    experimental::WorkUnitSpec wu{
        .name = "main", .kernels = names, .target_nodes = experimental::NodeCoord{logical_core}};
    experimental::ProgramSpec spec{.name = "watcher_ringbuf_multi", .kernels = specs, .work_units = {wu}};
    workload.add_program(device_range, experimental::MakeProgramFromSpec(*mesh_device, spec));

    const int dumps_before_run = MetalContext::instance().watcher_server()->dump_count();
    fixture->RunProgram(mesh_device, workload, true);

    if (expected_head) {
        const uint32_t max_head =
            *expected_head + (MetalContext::instance().watcher_server()->dump_count() - dumps_before_run);
        uint32_t head = 0;
        cluster.read_core(&head, sizeof(head), core, head_addr);
        EXPECT_GE(head, *expected_head);
        EXPECT_LE(head, max_head);
    }

    log_info(tt::LogTest, "Checking file: {}", fixture->log_file_name);
    EXPECT_TRUE(FileContainsAllStrings(fixture->log_file_name, expected));
}

struct RingBufferTestParams {
    std::string test_name;
    HalProcessorIdentifier processor;
};

class WatcherRingBufferTest : public MeshWatcherFixture, public ::testing::WithParamInterface<RingBufferTestParams> {};

TEST_P(WatcherRingBufferTest, TestWatcherRingBuffer) {
    const auto& params = GetParam();
    const auto& hal = MetalContext::instance().hal();
    const bool is_quasar = (hal.get_arch() == tt::ARCH::QUASAR);

    if (!hal.has_programmable_core_type(params.processor.core_type)) {
        GTEST_SKIP() << "Test " << params.test_name << ": core type not available on this architecture";
    }

    uint32_t available_processors = hal.get_processor_types_count(
        params.processor.core_type, static_cast<uint32_t>(params.processor.processor_class));
    if (params.processor.processor_type >= available_processors) {
        GTEST_SKIP() << "Test " << params.test_name << " requires processor type " << params.processor.processor_type
                     << " but only " << available_processors << " available on this architecture";
    }

    const bool is_reserved_quasar_dm = is_quasar && params.processor.core_type == HalProgrammableCoreType::TENSIX &&
                                       params.processor.processor_class == HalProcessorClassType::DM &&
                                       params.processor.processor_type < 2;
    if (is_reserved_quasar_dm) {
        GTEST_SKIP() << "Test " << params.test_name << ": DM0/DM1 are reserved for the runtime on Quasar";
    }

    const bool is_idle_eth = (params.processor.core_type == HalProgrammableCoreType::IDLE_ETH);
    const bool is_dram = (params.processor.core_type == HalProgrammableCoreType::DRAM);
    if ((is_idle_eth || is_dram) && !this->IsSlowDispatch()) {
        GTEST_SKIP() << "Test " << params.test_name << " requires Slow Dispatch";
    }

    const bool is_active_eth = (params.processor.core_type == HalProgrammableCoreType::ACTIVE_ETH);
    if (is_active_eth && !this->IsSlowDispatch()) {
        GTEST_SKIP() << "fast dispatch to active ethernet has been removed";
    }

    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [&params](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunTest(fixture, mesh_device, params.processor);
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
        RingBufferTestParams{"Brisc", {TENSIX, DM, 0}},
        RingBufferTestParams{"NCrisc", {TENSIX, DM, 1}},
        RingBufferTestParams{"DM2", {TENSIX, DM, 2}},
        RingBufferTestParams{"DM3", {TENSIX, DM, 3}},
        RingBufferTestParams{"DM4", {TENSIX, DM, 4}},
        RingBufferTestParams{"DM5", {TENSIX, DM, 5}},
        RingBufferTestParams{"DM6", {TENSIX, DM, 6}},
        RingBufferTestParams{"DM7", {TENSIX, DM, 7}},
        RingBufferTestParams{"Trisc0", {TENSIX, COMPUTE, 0}},
        RingBufferTestParams{"Trisc1", {TENSIX, COMPUTE, 1}},
        RingBufferTestParams{"Trisc2", {TENSIX, COMPUTE, 2}},
        RingBufferTestParams{"Trisc3", {TENSIX, COMPUTE, 3}},  // Quasar only
        RingBufferTestParams{"Erisc", {ACTIVE_ETH, DM, 0}},
        RingBufferTestParams{"IErisc", {IDLE_ETH, DM, 0}},
        RingBufferTestParams{"Drisc", {DRAM, DM, 0}}),
    [](const ::testing::TestParamInfo<RingBufferTestParams>& info) { return info.param.test_name; });

// Every writer on the core pushes from one program: 22 on Quasar (6 DMs + 16 TRISCs), 5 on
// Blackhole. The DM-vs-TRISC overlap is what the Quasar semaphore exists to serialize.
TEST_F(MeshWatcherFixture, TestWatcherRingBufferMpscMultiWriter) {
    const auto& hal = MetalContext::instance().hal();
    if (!hal.has_mpsc_ring_buffer()) {
        GTEST_SKIP() << "Multi-writer test requires the MPSC ring buffer";
    }
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunMultiWriterTest(fixture, mesh_device, false);
            },
            mesh_device);
    }
}

// All user processors on the core push across the point where push_to_ring_buffer() rewinds the head
// before its semaphore saturates.
TEST_F(MeshWatcherFixture, TestWatcherRingBufferMpscSemaphoreRewind) {
    if (MetalContext::instance().hal().get_arch() != tt::ARCH::QUASAR) {
        GTEST_SKIP() << "Only Quasar keeps the MPSC head in a saturating semaphore";
    }
    for (auto& mesh_device : this->devices_) {
        this->RunTestOnDevice(
            [](MeshWatcherFixture* fixture, const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
                RunMultiWriterTest(fixture, mesh_device, true);
            },
            mesh_device);
    }
}

}  // namespace
