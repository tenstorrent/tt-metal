// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole real-time-profiler sanity checks. Milestone 1 validates capability,
// nonblocking launch identity, and device start timestamps. Milestone 2 validates
// the device-local descriptor, completion-observer, reset, and loss protocol.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "hostdevcommon/common_values.hpp"
#include "hostdev/realtime_profiler_protocol_common.h"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/command_queue_common.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_settings.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"
#include "tt_metal/impl/program/program_impl.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "realtime_profiler_test_utils.hpp"

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::GetProgramRealtimeProfilerDeviceCapabilities;
using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeProfilerInactiveReason;
using tt::tt_metal::experimental::ProgramRealtimeRecord;
using tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

constexpr uint32_t kNumPrograms = 5;
// Generous upper bound: the inlined NOP loop kernels below run ~40K
// unrolled NOPs. Even on slow silicon that stays in the tens-of-microseconds
// range, so 1s is a sanity cap only intended to catch a broken clock /
// mis-decoded timestamp.

// Per-program marker embedded in the kernel source so the source-correlation
// assertion can verify each record carries the correct source.
constexpr const char* kSourceMarkerPrefix = "rt_profiler_marker_";

TEST(RealtimeProfilerProtocol, CompletionCounterComparisonHandlesNaturalWrap) {
    constexpr uint32_t kCounterWidth = 17;
    constexpr uint32_t kCounterMask = (1u << kCounterWidth) - 1;
    EXPECT_TRUE(realtime_profiler_modular_ge<kCounterWidth>(3, (kCounterMask - 2) & kCounterMask));
    EXPECT_TRUE(realtime_profiler_modular_ge<kCounterWidth>(0, kCounterMask));
    EXPECT_TRUE(realtime_profiler_modular_ge<kCounterWidth>(kCounterMask, kCounterMask));
    EXPECT_FALSE(realtime_profiler_modular_ge<kCounterWidth>((kCounterMask - 2) & kCounterMask, 3));
    EXPECT_FALSE(realtime_profiler_modular_ge<kCounterWidth>(kCounterMask, 0));
}

TEST(RealtimeProfilerProtocol, ProtectedGoWindowStagesOnlyLocalProfilerPayload) {
    std::filesystem::path root = std::filesystem::path(__FILE__).parent_path();
    const std::filesystem::path relative_source = "tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp";
    while (!root.empty() && !std::filesystem::exists(root / relative_source)) {
        const auto parent = root.parent_path();
        ASSERT_NE(parent, root) << "Could not locate repository root from " << __FILE__;
        root = parent;
    }
    std::ifstream input(root / relative_source);
    ASSERT_TRUE(input.good());
    const std::string source((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());

    const size_t stage_begin = source.find("void stage_realtime_profiler_start(");
    const size_t stage_end = source.find("void commit_realtime_profiler_start(", stage_begin);
    ASSERT_NE(stage_begin, std::string::npos);
    ASSERT_NE(stage_end, std::string::npos);
    const std::string_view stage_body(source.data() + stage_begin, stage_end - stage_begin);
    EXPECT_EQ(stage_body.find("NOC_STREAM_"), std::string_view::npos);
    EXPECT_EQ(stage_body.find("noc_"), std::string_view::npos);
    EXPECT_EQ(stage_body.find("invalidate_l1_cache"), std::string_view::npos);
    EXPECT_EQ(stage_body.find("descriptor_write_index"), std::string_view::npos);

    const size_t init_state = source.find("cq_noc_async_write_init_state<CQ_NOC_SNDL, true>");
    const size_t with_state = source.find("cq_noc_async_write_with_state<CQ_NOC_sndl, CQ_NOC_wait>", init_state);
    ASSERT_NE(init_state, std::string::npos);
    ASSERT_NE(with_state, std::string::npos);
    const std::string_view protected_window(source.data() + init_state, with_state - init_state);
    EXPECT_NE(protected_window.find("stage_realtime_profiler_start"), std::string_view::npos);
    EXPECT_NE(protected_window.find("wait_for_workers<false>"), std::string_view::npos);
    EXPECT_EQ(protected_window.find("commit_realtime_profiler_start"), std::string_view::npos);
}

TEST(RealtimeProfilerCapability, ReportsSingleCqArchitectureAndActivation) {
    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    auto* device = mesh_device->get_devices().front();
    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    EXPECT_EQ(capability->active, IsProgramRealtimeProfilerActive());

    auto& core_manager = MetalContext::instance(mesh_device->impl().get_context_id()).get_dispatch_core_manager();
    if (device->arch() == tt::ARCH::BLACKHOLE) {
        EXPECT_TRUE(capability->active);
        EXPECT_EQ(capability->inactive_reason, ProgramRealtimeProfilerInactiveReason::None);
        EXPECT_TRUE(core_manager.get_reserved_realtime_profiler_core(device->id()).has_value());
        const uint32_t maximum_completion_contribution =
            mesh_device->num_worker_cores(HalProgrammableCoreType::TENSIX, SubDeviceId{0}) +
            mesh_device->impl().num_virtual_eth_cores(SubDeviceId{0});
        EXPECT_LE(maximum_completion_contribution, std::numeric_limits<uint8_t>::max())
            << "The supported Blackhole topology must fit the validated go-command worker field";
    } else {
        EXPECT_FALSE(capability->active);
        EXPECT_EQ(capability->inactive_reason, ProgramRealtimeProfilerInactiveReason::UnsupportedArchitecture);
        EXPECT_FALSE(core_manager.get_reserved_realtime_profiler_core(device->id()).has_value());
    }

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerCapability, RejectsBlackholeMultipleHardwareCommandQueues) {
    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 2, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        mesh_device->close();
        GTEST_SKIP() << "The concurrent realtime profiler is Blackhole-only";
    }

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    ASSERT_EQ(capabilities.size(), 1u);
    EXPECT_FALSE(capabilities.front().active);
    EXPECT_EQ(
        capabilities.front().inactive_reason, ProgramRealtimeProfilerInactiveReason::MultipleHardwareCommandQueues);
    EXPECT_FALSE(IsProgramRealtimeProfilerActive());
    EXPECT_FALSE(MetalContext::instance(mesh_device->impl().get_context_id())
                     .get_dispatch_core_manager()
                     .get_reserved_realtime_profiler_core(device->id())
                     .has_value());
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerCapability, RejectsBlackholeEthDispatchTopology) {
    constexpr int kDeviceId = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    try {
        mesh_device = distributed::MeshDevice::create_unit_mesh(
            kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::ETH});
    } catch (const std::exception& e) {
        // Some Blackhole QuietBox descriptors reject ETH dispatch later in allocator setup. The profiler reservation
        // decision precedes that failure and must already have left the dispatch pool untouched.
        ASSERT_NE(std::string_view(e.what()).find("No core coordinate found"), std::string_view::npos) << e.what();
        EXPECT_FALSE(MetalContext::instance()
                         .get_dispatch_core_manager()
                         .get_reserved_realtime_profiler_core(kDeviceId)
                         .has_value());
        return;
    }
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        mesh_device->close();
        GTEST_SKIP() << "The concurrent realtime profiler is Blackhole-only";
    }

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    ASSERT_EQ(capabilities.size(), 1u);
    EXPECT_FALSE(capabilities.front().active);
    EXPECT_EQ(capabilities.front().inactive_reason, ProgramRealtimeProfilerInactiveReason::NonWorkerDispatch);
    EXPECT_FALSE(IsProgramRealtimeProfilerActive());
    EXPECT_FALSE(MetalContext::instance(mesh_device->impl().get_context_id())
                     .get_dispatch_core_manager()
                     .get_reserved_realtime_profiler_core(device->id())
                     .has_value());
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerCapability, EnvironmentDisableIsLatchedBeforeDispatchBuild) {
    const char* disabled = std::getenv("TT_METAL_DISABLE_REALTIME_PROFILER");
    if (disabled == nullptr || std::string_view(disabled) != "1") {
        GTEST_SKIP() << "Run this test in a fresh process with TT_METAL_DISABLE_REALTIME_PROFILER=1";
    }

    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    ASSERT_EQ(capabilities.size(), 1u);
    EXPECT_FALSE(capabilities.front().active);
    EXPECT_EQ(capabilities.front().inactive_reason, ProgramRealtimeProfilerInactiveReason::DisabledByEnvironment);
    EXPECT_FALSE(IsProgramRealtimeProfilerActive());
    EXPECT_FALSE(MetalContext::instance(mesh_device->impl().get_context_id())
                     .get_dispatch_core_manager()
                     .get_reserved_realtime_profiler_core(device->id())
                     .has_value());
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerProtocol, CleanBaselineL1BudgetAndScratchRegisterOwnership) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kProtocolMessageBudget = 8 * 1024;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (device->arch() != tt::ARCH::BLACKHOLE) {
        mesh_device->close();
        GTEST_SKIP() << "The clean-room register protocol is Blackhole-only";
    }

    auto& metal = MetalContext::instance(mesh_device->impl().get_context_id());
    const auto& dispatch_mem_map = metal.dispatch_mem_map();
    const uint32_t l1_size = metal.hal().get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::BASE);
    const uint32_t dispatch_end =
        dispatch_mem_map.dispatch_buffer_base(/*cq_id=*/0) +
        (dispatch_mem_map.dispatch_buffer_pages() << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE) +
        dispatch_mem_map.dispatch_s_buffer_size();
    ASSERT_LE(dispatch_end, l1_size);
    const uint32_t current_headroom = l1_size - dispatch_end;
    const uint32_t worst_case_aligned_growth = tt::align(
        kProtocolMessageBudget - sizeof(realtime_profiler_msg_t),
        1u << DispatchSettings::DISPATCH_BUFFER_LOG_PAGE_SIZE);

    EXPECT_GE(current_headroom, worst_case_aligned_growth)
        << "The clean-room profiler message budget no longer fits before the dispatch buffers reach the end of L1";
    log_info(
        tt::LogTest,
        "[RT profiler protocol] message={} B message_budget={} B dispatch_end=0x{:x} l1_size=0x{:x} "
        "headroom={} B worst_aligned_growth={} B",
        sizeof(realtime_profiler_msg_t),
        kProtocolMessageBudget,
        dispatch_end,
        l1_size,
        current_headroom,
        worst_case_aligned_growth);

    constexpr uint32_t kScratchMask = 0x00FFFFFF;
    constexpr uint32_t kScratch3TestValue = 0x0007E50A;
    constexpr uint32_t kScratch4TestValue = 0x005A1234;
    constexpr uint32_t kScratch5TestValue = 0x00A5FEDC;
    constexpr uint32_t kScratchAckValue = 0x003C0DE5;
    const CoreCoord worker{0, 0};
    const uint32_t output_addr =
        metal.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    std::vector<uint32_t> zeros(14, 0);
    detail::WriteToDeviceL1(device, worker, output_addr, zeros, CoreType::WORKER);

    Program scratch_program = CreateProgram();
    const std::string scratch_producer_kernel =
        "#include <cstdint>\n"
        "#include \"api/dataflow/dataflow_api.h\"\n"
        "void kernel_main() {\n"
        "  constexpr uint32_t stream = 8;\n"
        "  constexpr uint32_t scratch3 = 39;\n"
        "  constexpr uint32_t scratch4 = 40;\n"
        "  constexpr uint32_t scratch5 = 41;\n"
        "  constexpr uint32_t value3 = 0x0007E50A;\n"
        "  constexpr uint32_t value4 = 0x005A1234;\n"
        "  constexpr uint32_t value5 = 0x00A5FEDC;\n"
        "  constexpr uint32_t not_ready = 0x00102030;\n"
        "  constexpr uint32_t ack = 0x003C0DE5;\n"
        "  uint32_t old3 = NOC_STREAM_READ_REG(stream, scratch3);\n"
        "  uint32_t old4 = NOC_STREAM_READ_REG(stream, scratch4);\n"
        "  uint32_t old5 = NOC_STREAM_READ_REG(stream, scratch5);\n"
        "  NOC_STREAM_WRITE_REG(stream, scratch5, not_ready);\n"
        "  NOC_STREAM_WRITE_REG(stream, scratch3, value3);\n"
        "  NOC_STREAM_WRITE_REG(stream, scratch4, value4);\n"
        "  volatile tt_l1_ptr uint32_t* out =\n"
        "      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(0));\n"
        "  out[0] = old3; out[1] = old4; out[2] = old5;\n"
        "  out[3] = NOC_STREAM_READ_REG(stream, scratch3);\n"
        "  out[4] = NOC_STREAM_READ_REG(stream, scratch4);\n"
        "  asm volatile(\"fence w,w\" ::: \"memory\");\n"
        "  NOC_STREAM_WRITE_REG(stream, scratch5, value5);\n"
        "  out[5] = NOC_STREAM_READ_REG(stream, scratch5);\n"
        "  uint32_t observed_ack = 0;\n"
        "  for (uint32_t i = 0; i < 1000000; ++i) {\n"
        "    observed_ack = NOC_STREAM_READ_REG(stream, scratch4);\n"
        "    if (observed_ack == ack) break;\n"
        "  }\n"
        "  out[10] = observed_ack;\n"
        "  NOC_STREAM_WRITE_REG(stream, scratch3, old3);\n"
        "  NOC_STREAM_WRITE_REG(stream, scratch4, old4);\n"
        "  NOC_STREAM_WRITE_REG(stream, scratch5, old5);\n"
        "  out[11] = NOC_STREAM_READ_REG(stream, scratch3);\n"
        "  out[12] = NOC_STREAM_READ_REG(stream, scratch4);\n"
        "  out[13] = NOC_STREAM_READ_REG(stream, scratch5);\n"
        "}\n";
    const KernelHandle scratch_producer_handle = CreateKernelFromString(
        scratch_program,
        scratch_producer_kernel,
        worker,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    const std::string scratch_consumer_kernel =
        "#include <cstdint>\n"
        "#include \"api/compute/compute_kernel_api.h\"\n"
        "void kernel_main() {\n"
        "#if COMPILE_FOR_TRISC == 0\n"
        "  constexpr uint32_t stream = 8;\n"
        "  constexpr uint32_t scratch3 = 39;\n"
        "  constexpr uint32_t scratch4 = 40;\n"
        "  constexpr uint32_t scratch5 = 41;\n"
        "  constexpr uint32_t value3 = 0x0007E50A;\n"
        "  constexpr uint32_t value4 = 0x005A1234;\n"
        "  constexpr uint32_t value5 = 0x00A5FEDC;\n"
        "  constexpr uint32_t ack = 0x003C0DE5;\n"
        "  uint32_t observed3 = 0;\n"
        "  uint32_t observed4 = 0;\n"
        "  uint32_t observed5 = 0;\n"
        "  for (uint32_t i = 0; i < 1000000; ++i) {\n"
        "    observed3 = NOC_STREAM_READ_REG(stream, scratch3);\n"
        "    observed4 = NOC_STREAM_READ_REG(stream, scratch4);\n"
        "    observed5 = NOC_STREAM_READ_REG(stream, scratch5);\n"
        "    if (observed3 == value3 && observed4 == value4 && observed5 == value5) break;\n"
        "  }\n"
        "  volatile tt_l1_ptr uint32_t* out =\n"
        "      reinterpret_cast<volatile tt_l1_ptr uint32_t*>(" +
        std::to_string(output_addr) +
        "u);\n"
        "  out[6] = observed3; out[7] = observed4; out[8] = observed5;\n"
        "  if (observed3 == value3 && observed4 == value4 && observed5 == value5) {\n"
        "    NOC_STREAM_WRITE_REG(stream, scratch4, ack);\n"
        "    out[9] = ack;\n"
        "  }\n"
        "#endif\n"
        "}\n";
    CreateKernelFromString(scratch_program, scratch_consumer_kernel, worker, ComputeConfig{});
    SetRuntimeArgs(scratch_program, scratch_producer_handle, worker, {output_addr});
    distributed::MeshWorkload scratch_workload;
    scratch_workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(scratch_program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), scratch_workload, /*blocking=*/true);

    std::vector<uint32_t> scratch_result;
    detail::ReadFromDeviceL1(
        device, worker, output_addr, static_cast<uint32_t>(zeros.size() * sizeof(uint32_t)), scratch_result);
    ASSERT_EQ(scratch_result.size(), zeros.size());
    EXPECT_EQ(scratch_result[3] & kScratchMask, kScratch3TestValue);
    EXPECT_EQ(scratch_result[4] & kScratchMask, kScratch4TestValue);
    EXPECT_EQ(scratch_result[5] & kScratchMask, kScratch5TestValue);
    EXPECT_EQ(scratch_result[6] & kScratchMask, kScratch3TestValue);
    EXPECT_EQ(scratch_result[7] & kScratchMask, kScratch4TestValue);
    EXPECT_EQ(scratch_result[8] & kScratchMask, kScratch5TestValue);
    EXPECT_EQ(scratch_result[9] & kScratchMask, kScratchAckValue);
    EXPECT_EQ(scratch_result[10] & kScratchMask, kScratchAckValue);
    EXPECT_EQ(scratch_result[11], scratch_result[0]);
    EXPECT_EQ(scratch_result[12], scratch_result[1]);
    EXPECT_EQ(scratch_result[13], scratch_result[2]);

    EXPECT_TRUE(mesh_device->close());
}

// Inlined kernel source: 200 × 200 = 40K unrolled NOPs. Used for both data
// movement (BRISC/NCRISC) and compute (TRISC) RISCs. We inline rather than
// loading from a file under tt_metal/programming_examples/... because those
// files ship in the `metalium-examples` deb, while this test runs from
// `tt-metalium-validation` deb in CI (`metalium-basic-tests` job in
// merge-gate.yaml). Using CreateKernelFromString keeps the test
// self-contained and decoupled from install-rule changes. The 40K-NOP
// duration is the load-bearing property: it makes the implausible-duration
// check meaningful (a corrupted timestamp e.g. with swapped 32-bit halves
// would still satisfy end > start for ns-scale blank kernels but would
// surface here as a multi-second duration).
std::string make_sanity_kernel_source(uint32_t runtime_id) {
    return "#include <cstdint>\n"
           "// " +
           std::string(kSourceMarkerPrefix) + std::to_string(runtime_id) +
           "\n"
           "void kernel_main() {\n"
           "    for (int i = 0; i < 200; i++) {\n"
           "#pragma GCC unroll 65534\n"
           "        for (int j = 0; j < 200; j++) {\n"
           "            asm(\"nop\");\n"
           "        }\n"
           "    }\n"
           "}\n";
}

// Runs a single compute program on all tensix cores on `mesh_device`,
// tagged with `runtime_id`, so the RT profiler pipeline emits a record
// carrying that runtime_id (records with runtime_id == 0 are filtered
// out by the host-side receiver).
void enqueue_sanity_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t runtime_id, const CoreRange& all_cores) {
    Program program = CreateProgram();

    const std::string kernel_src = make_sanity_kernel_source(runtime_id);

    CreateKernelFromString(
        program,
        kernel_src,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernelFromString(
        program,
        kernel_src,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernelFromString(program, kernel_src, all_cores, ComputeConfig{});

    program.set_runtime_id(static_cast<uint64_t>(runtime_id));

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
}

distributed::MeshWorkload make_profiled_data_movement_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t runtime_id,
    const CoreRangeSet& cores,
    uint32_t loop_count) {
    Program program = CreateProgram();
    const std::string kernel_src =
        "#include <cstdint>\n"
        "void kernel_main() {\n"
        "  for (volatile uint32_t i = 0; i < " +
        std::to_string(loop_count) +
        "u; ++i) { asm volatile(\"nop\"); }\n"
        "}\n";
    CreateKernelFromString(
        program,
        kernel_src,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    program.set_runtime_id(static_cast<uint64_t>(runtime_id));
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    return workload;
}

void enqueue_profiled_data_movement_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t runtime_id,
    const CoreRangeSet& cores,
    uint32_t loop_count) {
    auto workload = make_profiled_data_movement_workload(mesh_device, runtime_id, cores, loop_count);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
}

struct RealtimeProfilerDispatchL1View {
    IDevice* device;
    CoreCoord dispatch_s_core;
    uint32_t profiler_base;
};

RealtimeProfilerDispatchL1View realtime_profiler_dispatch_l1_view(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto* device = mesh_device->get_devices().front();
    auto& metal = MetalContext::instance(mesh_device->impl().get_context_id());
    auto& core_manager = metal.get_dispatch_core_manager();
    const uint16_t channel = metal.get_cluster().get_assigned_channel_for_device(device->id());
    const tt_cxy_pair& dispatch_s_cxy = core_manager.dispatcher_s_core(device->id(), channel, 0);
    return {
        .device = device,
        .dispatch_s_core = CoreCoord{dispatch_s_cxy.x, dispatch_s_cxy.y},
        .profiler_base = metal.dispatch_mem_map().get_device_command_queue_addr(
            CommandQueueDeviceAddrType::REALTIME_PROFILER_MSG, 0),
    };
}

uint32_t wait_for_completed_records(
    const RealtimeProfilerDispatchL1View& view, uint32_t expected, std::chrono::steady_clock::duration timeout) {
    std::vector<uint32_t> completed_write(1, 0);
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    do {
        detail::ReadFromDeviceL1(
            view.device,
            view.dispatch_s_core,
            view.profiler_base + offsetof(realtime_profiler_msg_t, completed_write_index),
            sizeof(uint32_t),
            completed_write,
            CoreType::WORKER);
        if (completed_write[0] < expected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (completed_write[0] < expected && std::chrono::steady_clock::now() < deadline);
    return completed_write[0];
}

std::vector<uint32_t> read_completed_records(const RealtimeProfilerDispatchL1View& view, uint32_t count) {
    std::vector<uint32_t> records;
    detail::ReadFromDeviceL1(
        view.device,
        view.dispatch_s_core,
        view.profiler_base + offsetof(realtime_profiler_msg_t, completed_words),
        count * REALTIME_PROFILER_COMPLETED_RECORD_WORDS * sizeof(uint32_t),
        records,
        CoreType::WORKER);
    return records;
}

TEST(RealtimeProfilerSanity, FiveProgramsBackToBack) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    // Activation flips on during the init-sync handshake inside mesh open,
    // so this check is stable by the time create_unit_mesh returns. When it
    // returns false the RT profiler was disabled for this dispatch config
    // (ETH dispatch, non-MMIO chip, kernels nullified, no valid RT core) —
    // treat that as a graceful skip rather than a failure.
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    std::vector<ProgramRealtimeRecord> records;
    uint64_t dropped = 0;

    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&records, &dropped](const ProgramRealtimeRecordBatch& batch) {
            dropped += batch.dropped;
            records.insert(records.end(), batch.records.begin(), batch.records.end());
        });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});
    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        // Runtime IDs start at 1 so every program emits a record (runtime_id == 0
        // is reserved for infrastructure traffic and filtered host-side).
        enqueue_sanity_program(mesh_device, /*runtime_id=*/i + 1, all_cores);
    }

    mesh_device->quiesce_devices();
    // Give the receiver thread a bounded window to drain the final socket pages.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    UnregisterProgramRealtimeProfilerCallback(handle);

    ASSERT_GE(records.size(), kNumPrograms)
        << "Expected at least " << kNumPrograms << " RT profiler records (one per program), got " << records.size();
    EXPECT_EQ(dropped, 0u);

    std::vector<uint32_t> records_per_runtime_id(kNumPrograms + 1, 0);
    for (const auto& rec : records) {
        if (rec.runtime_id >= 1 && rec.runtime_id <= kNumPrograms) {
            ++records_per_runtime_id[rec.runtime_id];
        }
        EXPECT_GT(rec.start_timestamp, 0u);
        EXPECT_GT(rec.end_timestamp, rec.start_timestamp);
        EXPECT_EQ(rec.schema_version, REALTIME_PROFILER_RECORD_SCHEMA_VERSION);
        EXPECT_EQ(rec.record_type, REALTIME_PROFILER_RECORD_TYPE_INTERVAL);
        EXPECT_EQ(rec.command_queue_id, 0u);
        EXPECT_EQ(rec.dispatch_stream, 0u);
        EXPECT_EQ(rec.cumulative_source_dropped, 0u);
        EXPECT_GT(rec.frequency, 0.0) << "RT record frequency must be positive (runtime_id=" << rec.runtime_id
                                      << ", chip=" << rec.chip_id << ")";
    }
    for (uint32_t runtime_id = 1; runtime_id <= kNumPrograms; ++runtime_id) {
        EXPECT_EQ(records_per_runtime_id[runtime_id], 1u)
            << "Each GO must produce exactly one record; duplicate runtime_id=" << runtime_id
            << " can indicate that terminate published a stale buffer ID";
    }

    // Every program embeds "<prefix><runtime_id>" in its source, so we can verify each record carries the correct
    // source.
    std::set<uint32_t> programs_with_correct_sources;
    for (const auto& rec : records) {
        if (rec.runtime_id < 1 || rec.runtime_id > kNumPrograms) {
            continue;
        }
        ASSERT_FALSE(rec.kernel_sources.empty())
            << "RT record for runtime_id=" << rec.runtime_id << " carried no kernel sources";
        const std::string expected_marker = kSourceMarkerPrefix + std::to_string(rec.runtime_id);
        for (const auto& src : rec.kernel_sources) {
            EXPECT_NE(src.find(expected_marker), std::string_view::npos)
                << "RT record for runtime_id=" << rec.runtime_id << " carried the wrong program's source: " << src;
            EXPECT_EQ(src.find(kSourceMarkerPrefix), src.rfind(kSourceMarkerPrefix))
                << "RT record for runtime_id=" << rec.runtime_id << " carried more than one program marker";
        }
        programs_with_correct_sources.insert(rec.runtime_id);
    }
    EXPECT_EQ(programs_with_correct_sources.size(), kNumPrograms)
        << "Not every program's source was correctly correlated by runtime ID";
}

TEST(RealtimeProfilerConcurrentDevicePath, SameStreamRecordsCarryCorrelatedDeviceEndpoints) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kFirstRuntimeId = 0x7101;
    constexpr auto kObserverTimeout = std::chrono::seconds(2);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});
    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        enqueue_sanity_program(mesh_device, kFirstRuntimeId + i, all_cores);
    }
    distributed::Finish(mesh_device->mesh_command_queue());

    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    const uint32_t completed_write = wait_for_completed_records(l1_view, kNumPrograms, kObserverTimeout);
    ASSERT_GE(completed_write, kNumPrograms) << "TRISC0 did not publish the expected device-local completed records";

    const std::vector<uint32_t> records = read_completed_records(l1_view, kNumPrograms);
    ASSERT_EQ(records.size(), kNumPrograms * REALTIME_PROFILER_COMPLETED_RECORD_WORDS);

    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        const uint32_t* record = &records[i * REALTIME_PROFILER_COMPLETED_RECORD_WORDS];
        const uint64_t start_tick = (static_cast<uint64_t>(record[0]) << 32) | record[1];
        const uint64_t end_tick = (static_cast<uint64_t>(record[4]) << 32) | record[5];
        EXPECT_EQ(record[2] & 0xFFFF, kFirstRuntimeId + i);
        EXPECT_EQ(record[2] >> 16, 0u);
        EXPECT_EQ(record[3] & 0xFF, REALTIME_PROFILER_RECORD_SCHEMA_VERSION);
        EXPECT_EQ((record[3] >> 8) & 0xF, REALTIME_PROFILER_RECORD_TYPE_INTERVAL);
        EXPECT_EQ((record[3] >> 16) & 0xFF, 0u);
        EXPECT_GT(start_tick, 0u);
        EXPECT_GT(end_tick, start_tick);
        EXPECT_EQ(record[6], i + 1);
        EXPECT_EQ(record[7], 0u);
    }

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, ReverseStreamCompletionAndActualResetGeneration) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kSlowRuntimeId = 0x7201;
    constexpr uint32_t kFastRuntimeId = 0x7202;
    constexpr uint32_t kAfterResetRuntimeId = 0x7203;
    constexpr uint32_t kSecondFastRuntimeId = 0x7204;
    constexpr auto kObserverTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }
    if (mesh_device->compute_with_storage_grid_size().x < 2) {
        mesh_device->close();
        GTEST_SKIP() << "Two independent worker cores are required";
    }

    const CoreRangeSet stream0_cores(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    const CoreRangeSet stream1_cores(CoreRange(CoreCoord{1, 0}, CoreCoord{1, 0}));
    const SubDevice stream0(std::array{stream0_cores});
    const SubDevice stream1(std::array{stream1_cores});
    const auto sub_device_manager = mesh_device->create_sub_device_manager({stream0, stream1}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto slow_workload = make_profiled_data_movement_workload(mesh_device, kSlowRuntimeId, stream0_cores, 100'000'000);
    auto fast_workload = make_profiled_data_movement_workload(mesh_device, kFastRuntimeId, stream1_cores, 5'000'000);
    auto second_fast_workload =
        make_profiled_data_movement_workload(mesh_device, kSecondFastRuntimeId, stream1_cores, 5'000'000);
    // Commit both binaries before the ordering arm so dispatch binary writes do
    // not delay the second GO long enough to hide the intended overlap.
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), slow_workload, /*blocking=*/true);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), fast_workload, /*blocking=*/true);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), second_fast_workload, /*blocking=*/true);
    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    ASSERT_GE(wait_for_completed_records(l1_view, 3, kObserverTimeout), 3u);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), slow_workload, /*blocking=*/false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), fast_workload, /*blocking=*/false);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), second_fast_workload, /*blocking=*/false);
    distributed::Finish(mesh_device->mesh_command_queue());

    ASSERT_GE(wait_for_completed_records(l1_view, 6, kObserverTimeout), 6u);
    auto records = read_completed_records(l1_view, 6);
    ASSERT_EQ(records.size(), 6 * REALTIME_PROFILER_COMPLETED_RECORD_WORDS);

    uint32_t slow_position = std::numeric_limits<uint32_t>::max();
    uint32_t fast_position = std::numeric_limits<uint32_t>::max();
    uint32_t second_fast_position = std::numeric_limits<uint32_t>::max();
    uint32_t slow_generation = 0;
    for (uint32_t i = 3; i < 6; ++i) {
        const uint32_t* record = &records[i * REALTIME_PROFILER_COMPLETED_RECORD_WORDS];
        const uint32_t runtime_id = record[2] & 0xFFFF;
        if (runtime_id == kSlowRuntimeId) {
            slow_position = i;
            slow_generation = record[2] >> 16;
            EXPECT_EQ((record[3] >> 16) & 0xFF, 0u);
        } else if (runtime_id == kFastRuntimeId) {
            fast_position = i;
            EXPECT_EQ((record[3] >> 16) & 0xFF, 1u);
        } else if (runtime_id == kSecondFastRuntimeId) {
            second_fast_position = i;
            EXPECT_EQ((record[3] >> 16) & 0xFF, 1u);
        }
    }
    ASSERT_NE(slow_position, std::numeric_limits<uint32_t>::max());
    ASSERT_NE(fast_position, std::numeric_limits<uint32_t>::max());
    ASSERT_NE(second_fast_position, std::numeric_limits<uint32_t>::max());
    EXPECT_LT(fast_position, slow_position)
        << "The completed ring must reflect independent stream completion, not submission order";
    EXPECT_LT(second_fast_position, slow_position);
    EXPECT_LT(fast_position, second_fast_position)
        << "Back-to-back invocations on one stream must retain invocation order while another stream is active";

    // Exercise the production CLEAR_STREAM lifecycle by returning to the default
    // sub-device manager, then prove the next descriptor carries the new epoch.
    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
    enqueue_profiled_data_movement_program(mesh_device, kAfterResetRuntimeId, stream0_cores, 1);
    distributed::Finish(mesh_device->mesh_command_queue());
    ASSERT_GE(wait_for_completed_records(l1_view, 7, kObserverTimeout), 7u);
    records = read_completed_records(l1_view, 7);

    bool found_after_reset = false;
    for (uint32_t i = 0; i < 7; ++i) {
        const uint32_t* record = &records[i * REALTIME_PROFILER_COMPLETED_RECORD_WORDS];
        if ((record[2] & 0xFFFF) == kAfterResetRuntimeId) {
            found_after_reset = true;
            EXPECT_NE(record[2] >> 16, slow_generation);
            EXPECT_EQ((record[3] >> 16) & 0xFF, 0u);
            EXPECT_EQ((record[3] >> 8) & 0xF, REALTIME_PROFILER_RECORD_TYPE_INTERVAL);
        }
    }
    EXPECT_TRUE(found_after_reset);

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, NextLaunchCannotCrossClearToGenerationWindow) {
    if (std::getenv("TT_METAL_REALTIME_PROFILER_TEST_RESET_PAUSE") == nullptr) {
        GTEST_SKIP() << "Run this test in a fresh process with TT_METAL_REALTIME_PROFILER_TEST_RESET_PAUSE=1";
    }

    constexpr int kDeviceId = 0;
    constexpr uint32_t kBeforeResetRuntimeId = 0x7251;
    constexpr uint32_t kAfterResetRuntimeId = 0x7252;
    constexpr auto kTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    const CoreRangeSet one_core(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    const auto sub_device_manager = mesh_device->create_sub_device_manager({SubDevice(std::array{one_core})}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);
    auto before_workload = make_profiled_data_movement_workload(mesh_device, kBeforeResetRuntimeId, one_core, 1);
    auto after_workload = make_profiled_data_movement_workload(mesh_device, kAfterResetRuntimeId, one_core, 1);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), before_workload, /*blocking=*/true);

    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    ASSERT_GE(wait_for_completed_records(l1_view, 1, kTimeout), 1u);
    std::vector<uint32_t> before_state;
    detail::ReadFromDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, stream_generation),
        sizeof(uint32_t),
        before_state,
        CoreType::WORKER);
    ASSERT_EQ(before_state.size(), 1u);

    std::vector<uint32_t> pause_request = {1};
    detail::WriteToDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, test_protocol_words),
        pause_request,
        CoreType::WORKER);
    // This queues the real WAIT_STREAM|CLEAR_STREAM lifecycle transaction and
    // returns while dispatch_d is held immediately after the counter clear.
    mesh_device->clear_loaded_sub_device_manager();

    bool pause_observed = false;
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    do {
        std::vector<uint32_t> observed;
        detail::ReadFromDeviceL1(
            device,
            l1_view.dispatch_s_core,
            l1_view.profiler_base + offsetof(realtime_profiler_msg_t, test_protocol_words),
            sizeof(uint32_t),
            observed,
            CoreType::WORKER);
        pause_observed = !observed.empty() && observed[0] == 2;
        if (!pause_observed) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (!pause_observed && std::chrono::steady_clock::now() < deadline);

    if (pause_observed) {
        // Put a real GO behind the paused reset transaction. Its descriptor and
        // completion must not become visible before generation publication.
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), after_workload, /*blocking=*/false);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::vector<uint32_t> paused_state;
        detail::ReadFromDeviceL1(
            device,
            l1_view.dispatch_s_core,
            l1_view.profiler_base + offsetof(realtime_profiler_msg_t, stream_generation),
            sizeof(uint32_t),
            paused_state,
            CoreType::WORKER);
        EXPECT_EQ(paused_state, before_state);
        EXPECT_EQ(wait_for_completed_records(l1_view, 2, std::chrono::milliseconds(20)), 1u);
    }

    // Always release the test hook before asserting, so a failure cannot leave
    // dispatch firmware parked during teardown.
    pause_request[0] = 0;
    detail::WriteToDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, test_protocol_words),
        pause_request,
        CoreType::WORKER);
    ASSERT_TRUE(pause_observed);
    distributed::Finish(mesh_device->mesh_command_queue());
    ASSERT_GE(wait_for_completed_records(l1_view, 2, kTimeout), 2u);

    const auto records = read_completed_records(l1_view, 2);
    bool found_after_reset = false;
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t* record = &records[i * REALTIME_PROFILER_COMPLETED_RECORD_WORDS];
        if ((record[2] & 0xFFFF) == kAfterResetRuntimeId) {
            found_after_reset = true;
            EXPECT_NE(record[2] >> 16, before_state[0] & 0xFFFF);
            EXPECT_EQ((record[3] >> 8) & 0xF, REALTIME_PROFILER_RECORD_TYPE_INTERVAL);
        }
    }
    EXPECT_TRUE(found_after_reset);

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    EXPECT_EQ(capability->loss.reset_descriptor, 0u);

    mesh_device->remove_sub_device_manager(sub_device_manager);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, CompletedQueueTransportDrainsWithoutBlockingDispatch) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kRuntimeId = 0x7301;
    constexpr uint32_t kPressureLaunches = REALTIME_PROFILER_COMPLETED_QUEUE_CAPACITY + 8;
    constexpr auto kTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    const CoreRangeSet one_core(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    auto workload = make_profiled_data_movement_workload(mesh_device, kRuntimeId, one_core, 1);
    test_utils::RealtimeProfilerRecordCollector collector;
    const auto callback = RegisterProgramRealtimeProfilerCallback(
        [&collector](const ProgramRealtimeRecordBatch& batch) { collector.consume(batch); });

    // Commit the binary first, then exceed the completed-ring capacity. Active
    // draining must recycle slots without waiting in the application path.
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
    for (uint32_t i = 0; i < kPressureLaunches; ++i) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
    }
    // This application completion is the nonblocking invariant under profiler pressure.
    distributed::Finish(mesh_device->mesh_command_queue());

    const auto callback_result = collector.wait_for_record_count(kRuntimeId, kPressureLaunches + 1, kTimeout);
    UnregisterProgramRealtimeProfilerCallback(callback);
    EXPECT_TRUE(callback_result.complete);
    EXPECT_EQ(callback_result.host_dropped, 0u);

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    EXPECT_EQ(capability->loss.completed_record, 0u);
    EXPECT_EQ(capability->loss.device_ring, 0u);

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, DeviceRingLossReachesCapabilityAndCallbackAggregate) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kDroppedRuntimeId = 0x7381;
    constexpr uint32_t kSentinelRuntimeId = 0x7382;
    constexpr auto kTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active";
    }

    auto& metal = MetalContext::instance(mesh_device->impl().get_context_id());
    const auto profiler_core = metal.get_dispatch_core_manager().get_reserved_realtime_profiler_core(device->id());
    ASSERT_TRUE(profiler_core.has_value());
    const uint32_t ring_addr =
        metal.dispatch_mem_map().get_device_command_queue_addr(CommandQueueDeviceAddrType::UNRESERVED, 0);

    const CoreRangeSet one_core(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    auto dropped_workload = make_profiled_data_movement_workload(mesh_device, kDroppedRuntimeId, one_core, 1);
    auto sentinel_workload = make_profiled_data_movement_workload(mesh_device, kSentinelRuntimeId, one_core, 1);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), dropped_workload, /*blocking=*/true);

    test_utils::RealtimeProfilerRecordCollector collector;
    const auto callback = RegisterProgramRealtimeProfilerCallback(
        [&collector](const ProgramRealtimeRecordBatch& batch) { collector.consume(batch); });

    // Test-only pressure injection preserves the consumer-owned read index.
    // Zero every published slot first so the real NCRISC drain cannot decode
    // stale L1 as phantom profiler records.
    std::vector<uint32_t> zero_ring_data(RT_PROFILER_RING_CAPACITY * RT_PROFILER_ENTRY_SIZE / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(
        device, *profiler_core, ring_addr + offsetof(RtProfilerRingBuffer, data), zero_ring_data, CoreType::WORKER);
    std::vector<uint32_t> ring_indices;
    detail::ReadFromDeviceL1(device, *profiler_core, ring_addr, 2 * sizeof(uint32_t), ring_indices, CoreType::WORKER);
    ASSERT_EQ(ring_indices.size(), 2u);
    const uint32_t injected_write_index = ring_indices[1] + RT_PROFILER_HOST_FIFO_PAGES;
    std::vector<uint32_t> pressure_write_index = {injected_write_index};
    detail::WriteToDeviceL1(
        device,
        *profiler_core,
        ring_addr + offsetof(RtProfilerRingBuffer, write_index),
        pressure_write_index,
        CoreType::WORKER);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), dropped_workload, /*blocking=*/true);

    uint64_t device_ring_loss = 0;
    const auto loss_deadline = std::chrono::steady_clock::now() + kTimeout;
    do {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
            return entry.chip_id == static_cast<uint32_t>(device->id());
        });
        ASSERT_NE(capability, capabilities.end());
        device_ring_loss = capability->loss.device_ring;
        if (device_ring_loss == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (device_ring_loss == 0 && std::chrono::steady_clock::now() < loss_deadline);
    ASSERT_GT(device_ring_loss, 0u);

    // NCRISC publishes read_index only after its whole reserved batch has been
    // pushed. Never rewrite write_index while that drain may be in flight;
    // wait boundedly for the consumer to reach the injected producer index.
    bool ring_empty = false;
    const auto drain_deadline = std::chrono::steady_clock::now() + kTimeout;
    do {
        detail::ReadFromDeviceL1(
            device, *profiler_core, ring_addr, 2 * sizeof(uint32_t), ring_indices, CoreType::WORKER);
        ring_empty = ring_indices[0] == injected_write_index && ring_indices[1] == injected_write_index;
        if (!ring_empty) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while (!ring_empty && std::chrono::steady_clock::now() < drain_deadline);
    ASSERT_TRUE(ring_empty);

    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), sentinel_workload, /*blocking=*/false);
    distributed::Finish(mesh_device->mesh_command_queue());
    const auto callback_result = collector.wait_for_record_count(kSentinelRuntimeId, 1, kTimeout);
    UnregisterProgramRealtimeProfilerCallback(callback);
    ASSERT_TRUE(callback_result.complete);

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    EXPECT_EQ(capability->loss.device_ring, device_ring_loss);

    const auto records = collector.records();
    const auto sentinel_record = std::find_if(
        records.begin(), records.end(), [](const auto& record) { return record.runtime_id == kSentinelRuntimeId; });
    ASSERT_NE(sentinel_record, records.end());
    EXPECT_EQ(sentinel_record->cumulative_source_dropped, device_ring_loss);
    const auto device_loss = collector.device_loss();
    ASSERT_FALSE(device_loss.empty());
    EXPECT_EQ(device_loss.front().cumulative_source_dropped, device_ring_loss);

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, FullDescriptorRingCountsDropAndStuckHeadWithoutBlocking) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kInjectedStream = REALTIME_PROFILER_MAX_STREAMS - 1;
    constexpr uint32_t kDroppedRuntimeId = 0x7401;
    constexpr uint32_t kWakeRuntimeId = 0x7402;
    constexpr uint32_t kSentinelRuntimeId = 0x7403;
    constexpr auto kLossTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }
    if (mesh_device->compute_with_storage_grid_size().x < REALTIME_PROFILER_MAX_STREAMS) {
        mesh_device->close();
        GTEST_SKIP() << "Eight independent worker cores are required";
    }

    std::vector<CoreRangeSet> stream_cores;
    std::vector<SubDevice> sub_devices;
    stream_cores.reserve(REALTIME_PROFILER_MAX_STREAMS);
    sub_devices.reserve(REALTIME_PROFILER_MAX_STREAMS);
    for (uint32_t i = 0; i < REALTIME_PROFILER_MAX_STREAMS; ++i) {
        stream_cores.emplace_back(CoreRange(CoreCoord{i, 0}, CoreCoord{i, 0}));
        sub_devices.emplace_back(std::array{stream_cores.back()});
    }
    const auto sub_device_manager = mesh_device->create_sub_device_manager(sub_devices, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    std::vector<uint32_t> generation(1, 0);
    detail::ReadFromDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, stream_generation) +
            kInjectedStream * sizeof(uint32_t),
        sizeof(uint32_t),
        generation,
        CoreType::WORKER);

    // Test-only fault injection: publish a full stream-7 ring in L1. A real
    // descriptor publication on stream 0 supplies the register-space wakeup;
    // production firmware still owns every normal producer transition.
    std::vector<uint32_t> injected_descriptors(
        REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY * REALTIME_PROFILER_DESCRIPTOR_WORDS, 0);
    for (uint32_t i = 0; i < REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY; ++i) {
        uint32_t* descriptor = &injected_descriptors[i * REALTIME_PROFILER_DESCRIPTOR_WORDS];
        descriptor[0] = 0x74F0 + i;
        descriptor[3] = 0xFFFF;
        descriptor[4] = generation[0];
    }
    const uint32_t injected_descriptor_addr = l1_view.profiler_base +
                                              offsetof(realtime_profiler_msg_t, descriptor_words) +
                                              kInjectedStream * REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY *
                                                  REALTIME_PROFILER_DESCRIPTOR_WORDS * sizeof(uint32_t);
    detail::WriteToDeviceL1(
        device, l1_view.dispatch_s_core, injected_descriptor_addr, injected_descriptors, CoreType::WORKER);
    std::vector<uint32_t> injected_write_index = {REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY};
    detail::WriteToDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, descriptor_write_index) +
            kInjectedStream * sizeof(uint32_t),
        injected_write_index,
        CoreType::WORKER);

    enqueue_profiled_data_movement_program(
        mesh_device, kDroppedRuntimeId, stream_cores[kInjectedStream], /*loop_count=*/1);
    enqueue_profiled_data_movement_program(mesh_device, kWakeRuntimeId, stream_cores[0], /*loop_count=*/1);
    distributed::Finish(mesh_device->mesh_command_queue());

    uint64_t descriptor_full = 0;
    uint64_t stuck_head = 0;
    const auto deadline = std::chrono::steady_clock::now() + kLossTimeout;
    do {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
            return entry.chip_id == static_cast<uint32_t>(device->id());
        });
        ASSERT_NE(capability, capabilities.end());
        descriptor_full = capability->loss.descriptor_full;
        stuck_head = capability->loss.stuck_head;
        if (descriptor_full == 0 || stuck_head == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while ((descriptor_full == 0 || stuck_head == 0) && std::chrono::steady_clock::now() < deadline);

    EXPECT_EQ(descriptor_full, 1u);
    EXPECT_EQ(stuck_head, 1u);

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);

    // Publish one ordinary sentinel after pressure and reset recovery. The
    // callback-visible aggregate must match the detailed diagnostic snapshot;
    // callback users must not need diagnostic L1 access to detect device loss.
    test_utils::RealtimeProfilerRecordCollector collector;
    const auto callback = RegisterProgramRealtimeProfilerCallback(
        [&collector](const ProgramRealtimeRecordBatch& batch) { collector.consume(batch); });
    const CoreRangeSet one_core(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    auto sentinel = make_profiled_data_movement_workload(mesh_device, kSentinelRuntimeId, one_core, 1);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), sentinel, /*blocking=*/false);
    distributed::Finish(mesh_device->mesh_command_queue());
    const auto callback_result = collector.wait_for_record_count(kSentinelRuntimeId, 1, kLossTimeout);
    UnregisterProgramRealtimeProfilerCallback(callback);
    ASSERT_TRUE(callback_result.complete);
    EXPECT_EQ(callback_result.host_dropped, 0u);

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    const uint64_t diagnostic_aggregate =
        capability->loss.descriptor_full + capability->loss.unsupported_launch + capability->loss.reset_descriptor +
        capability->loss.observer_coalesced + capability->loss.completed_record + capability->loss.terminal_descriptor +
        capability->loss.terminal_record + capability->loss.observer_stop_timeout + capability->loss.device_ring;
    ASSERT_GT(diagnostic_aggregate, 0u);

    const auto callback_records = collector.records();
    const auto sentinel_record = std::find_if(callback_records.begin(), callback_records.end(), [](const auto& record) {
        return record.runtime_id == kSentinelRuntimeId;
    });
    ASSERT_NE(sentinel_record, callback_records.end());
    EXPECT_EQ(sentinel_record->cumulative_source_dropped, diagnostic_aggregate);

    const auto callback_device_loss = collector.device_loss();
    const auto callback_snapshot =
        std::find_if(callback_device_loss.begin(), callback_device_loss.end(), [device](const auto& snapshot) {
            return snapshot.chip_id == static_cast<uint32_t>(device->id());
        });
    ASSERT_NE(callback_snapshot, callback_device_loss.end());
    EXPECT_EQ(callback_snapshot->cumulative_source_dropped, diagnostic_aggregate);

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, CoalescedSatisfiedDescriptorsKeepNewestAndCountExactLoss) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kInjectedStream = REALTIME_PROFILER_MAX_STREAMS - 1;
    constexpr uint32_t kFirstInjectedRuntimeId = 0x7500;
    constexpr uint32_t kWakeRuntimeId = 0x7510;
    constexpr auto kObserverTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    std::vector<uint32_t> generation(1, 0);
    detail::ReadFromDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, stream_generation) +
            kInjectedStream * sizeof(uint32_t),
        sizeof(uint32_t),
        generation,
        CoreType::WORKER);

    std::vector<uint32_t> injected_descriptors(
        REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY * REALTIME_PROFILER_DESCRIPTOR_WORDS, 0);
    for (uint32_t i = 0; i < REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY; ++i) {
        uint32_t* descriptor = &injected_descriptors[i * REALTIME_PROFILER_DESCRIPTOR_WORDS];
        descriptor[0] = kFirstInjectedRuntimeId + i;
        descriptor[2] = 100 + i;
        descriptor[3] = 0;  // already satisfied by the initial stream count
        descriptor[4] = generation[0];
    }
    detail::WriteToDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, descriptor_words) +
            kInjectedStream * REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY * REALTIME_PROFILER_DESCRIPTOR_WORDS *
                sizeof(uint32_t),
        injected_descriptors,
        CoreType::WORKER);
    std::vector<uint32_t> injected_write_index = {REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY};
    detail::WriteToDeviceL1(
        device,
        l1_view.dispatch_s_core,
        l1_view.profiler_base + offsetof(realtime_profiler_msg_t, descriptor_write_index) +
            kInjectedStream * sizeof(uint32_t),
        injected_write_index,
        CoreType::WORKER);

    const CoreRangeSet one_core(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    enqueue_profiled_data_movement_program(mesh_device, kWakeRuntimeId, one_core, /*loop_count=*/1);
    distributed::Finish(mesh_device->mesh_command_queue());
    ASSERT_GE(wait_for_completed_records(l1_view, 2, kObserverTimeout), 2u);

    const auto records = read_completed_records(l1_view, 2);
    uint32_t injected_record_count = 0;
    uint32_t injected_runtime_id = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t* record = &records[i * REALTIME_PROFILER_COMPLETED_RECORD_WORDS];
        if (((record[3] >> 16) & 0xFF) == kInjectedStream) {
            ++injected_record_count;
            injected_runtime_id = record[2] & 0xFFFF;
        }
    }
    EXPECT_EQ(injected_record_count, 1u);
    EXPECT_EQ(injected_runtime_id, kFirstInjectedRuntimeId + REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY - 1);

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    EXPECT_EQ(capability->loss.observer_coalesced, REALTIME_PROFILER_DESCRIPTOR_QUEUE_CAPACITY - 1);

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, DeviceClockBoundsGoLeadAndObserverPollingLag) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kRuntimeId = 0x7601;
    constexpr uint32_t kSamples = 32;
    constexpr auto kObserverTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    const CoreCoord worker{0, 0};
    auto& metal = MetalContext::instance(mesh_device->impl().get_context_id());
    const uint32_t output_addr =
        metal.hal().get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::DEFAULT_UNRESERVED);
    std::vector<uint32_t> zero_output(4, 0);
    detail::WriteToDeviceL1(device, worker, output_addr, zero_output, CoreType::WORKER);

    Program program = CreateProgram();
    const std::string kernel_src =
        "#include <cstdint>\n"
        "#include \"api/dataflow/dataflow_api.h\"\n"
        "#include \"risc_common.h\"\n"
        "void kernel_main() {\n"
        "  volatile tt_reg_ptr uint32_t* clock = reinterpret_cast<volatile tt_reg_ptr uint32_t*>("
        "RISCV_DEBUG_REG_WALL_CLOCK_L);\n"
        "  volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>("
        "get_arg_val<uint32_t>(0));\n"
        "  uint32_t start_lo = clock[0]; uint32_t start_hi = clock[2];\n"
        "  out[0] = start_hi; out[1] = start_lo;\n"
        "  for (volatile uint32_t i = 0; i < 10000; ++i) { asm volatile(\"nop\"); }\n"
        "  uint32_t end_lo = clock[0]; uint32_t end_hi = clock[2];\n"
        "  out[2] = end_hi; out[3] = end_lo;\n"
        "}\n";
    const KernelHandle kernel = CreateKernelFromString(
        program,
        kernel_src,
        worker,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, worker, {output_addr});
    program.set_runtime_id(kRuntimeId);
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));

    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    std::vector<uint64_t> go_lead_cycles;
    std::vector<uint64_t> observer_lag_cycles;
    go_lead_cycles.reserve(kSamples);
    observer_lag_cycles.reserve(kSamples);
    for (uint32_t i = 0; i < kSamples; ++i) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
        ASSERT_GE(wait_for_completed_records(l1_view, i + 1, kObserverTimeout), i + 1);

        std::vector<uint32_t> worker_ticks;
        detail::ReadFromDeviceL1(device, worker, output_addr, 4 * sizeof(uint32_t), worker_ticks, CoreType::WORKER);
        ASSERT_EQ(worker_ticks.size(), 4u);
        std::vector<uint32_t> record;
        detail::ReadFromDeviceL1(
            device,
            l1_view.dispatch_s_core,
            l1_view.profiler_base + offsetof(realtime_profiler_msg_t, completed_words) +
                i * REALTIME_PROFILER_COMPLETED_RECORD_WORDS * sizeof(uint32_t),
            REALTIME_PROFILER_COMPLETED_RECORD_WORDS * sizeof(uint32_t),
            record,
            CoreType::WORKER);
        ASSERT_EQ(record.size(), REALTIME_PROFILER_COMPLETED_RECORD_WORDS);

        const uint64_t descriptor_start = (static_cast<uint64_t>(record[0]) << 32) | record[1];
        const uint64_t worker_start = (static_cast<uint64_t>(worker_ticks[0]) << 32) | worker_ticks[1];
        const uint64_t worker_end = (static_cast<uint64_t>(worker_ticks[2]) << 32) | worker_ticks[3];
        const uint64_t observer_end = (static_cast<uint64_t>(record[4]) << 32) | record[5];
        ASSERT_GE(worker_start, descriptor_start);
        ASSERT_GE(observer_end, worker_end);
        go_lead_cycles.push_back(worker_start - descriptor_start);
        observer_lag_cycles.push_back(observer_end - worker_end);
    }

    std::sort(go_lead_cycles.begin(), go_lead_cycles.end());
    std::sort(observer_lag_cycles.begin(), observer_lag_cycles.end());
    const size_t p95_index = (95 * kSamples + 99) / 100 - 1;
    log_info(
        tt::LogTest,
        "[RT profiler M2 device cycles] go_lead min={} median={} p95={} max={}; observer_lag min={} median={} "
        "p95={} max={}",
        go_lead_cycles.front(),
        go_lead_cycles[kSamples / 2],
        go_lead_cycles[p95_index],
        go_lead_cycles.back(),
        observer_lag_cycles.front(),
        observer_lag_cycles[kSamples / 2],
        observer_lag_cycles[p95_index],
        observer_lag_cycles.back());
    EXPECT_LT(go_lead_cycles.back(), 10'000'000u);
    EXPECT_LT(observer_lag_cycles.back(), 10'000'000u);

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, MeasuresIdleOneAndAllStreamObserverCycles) {
    if (std::getenv("TT_METAL_REALTIME_PROFILER_TEST_OBSERVER_CYCLES") == nullptr) {
        GTEST_SKIP() << "Run this test in a fresh process with TT_METAL_REALTIME_PROFILER_TEST_OBSERVER_CYCLES=1";
    }

    constexpr int kDeviceId = 0;
    constexpr auto kTimeout = std::chrono::seconds(3);
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }
    if (mesh_device->compute_with_storage_grid_size().x < REALTIME_PROFILER_MAX_STREAMS) {
        mesh_device->close();
        GTEST_SKIP() << "Eight independent worker cores are required";
    }

    std::vector<CoreRangeSet> stream_cores;
    std::vector<SubDevice> sub_devices;
    std::vector<distributed::MeshWorkload> workloads;
    stream_cores.reserve(REALTIME_PROFILER_MAX_STREAMS);
    sub_devices.reserve(REALTIME_PROFILER_MAX_STREAMS);
    workloads.reserve(REALTIME_PROFILER_MAX_STREAMS);
    for (uint32_t stream = 0; stream < REALTIME_PROFILER_MAX_STREAMS; ++stream) {
        stream_cores.emplace_back(CoreRange(CoreCoord{stream, 0}, CoreCoord{stream, 0}));
        sub_devices.emplace_back(std::array{stream_cores.back()});
    }
    const auto sub_device_manager = mesh_device->create_sub_device_manager(sub_devices, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);
    for (uint32_t stream = 0; stream < REALTIME_PROFILER_MAX_STREAMS; ++stream) {
        workloads.emplace_back(
            make_profiled_data_movement_workload(mesh_device, 0x7900 + stream, stream_cores[stream], 50'000'000));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workloads.back(), /*blocking=*/true);
    }

    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workloads[0], /*blocking=*/true);
    for (uint32_t stream = 0; stream < REALTIME_PROFILER_MAX_STREAMS; ++stream) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workloads[stream], /*blocking=*/false);
    }
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<uint32_t> cycle_data;
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    do {
        detail::ReadFromDeviceL1(
            l1_view.device,
            l1_view.dispatch_s_core,
            l1_view.profiler_base + offsetof(realtime_profiler_msg_t, test_protocol_words),
            6 * sizeof(uint32_t),
            cycle_data,
            CoreType::WORKER);
        if (cycle_data.size() != 6 || cycle_data[3] == 0 || cycle_data[4] == 0 || cycle_data[5] == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while ((cycle_data.size() != 6 || cycle_data[3] == 0 || cycle_data[4] == 0 || cycle_data[5] == 0) &&
             std::chrono::steady_clock::now() < deadline);
    ASSERT_EQ(cycle_data.size(), 6u);
    ASSERT_GT(cycle_data[3], 0u);
    ASSERT_GT(cycle_data[4], 0u);
    ASSERT_GT(cycle_data[5], 0u);
    log_info(
        tt::LogTest,
        "[RT profiler M2 observer cycles, instrumented upper bound] idle={} (n={}), one_stream={} (n={}), "
        "all_streams={} (n={})",
        cycle_data[0] / cycle_data[3],
        cycle_data[3],
        cycle_data[1] / cycle_data[4],
        cycle_data[4],
        cycle_data[2] / cycle_data[5],
        cycle_data[5]);

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, DepthFourAvoidsDescriptorLossForShortestTwoStreamLaunches) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kLaunchesPerStream = 256;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }
    if (mesh_device->compute_with_storage_grid_size().x < 2) {
        mesh_device->close();
        GTEST_SKIP() << "Two independent worker cores are required";
    }

    const CoreRangeSet stream0_cores(CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}));
    const CoreRangeSet stream1_cores(CoreRange(CoreCoord{1, 0}, CoreCoord{1, 0}));
    const auto sub_device_manager = mesh_device->create_sub_device_manager(
        {SubDevice(std::array{stream0_cores}), SubDevice(std::array{stream1_cores})}, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);

    auto stream0_workload = make_profiled_data_movement_workload(mesh_device, 0x7701, stream0_cores, 1);
    auto stream1_workload = make_profiled_data_movement_workload(mesh_device, 0x7702, stream1_cores, 1);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), stream0_workload, /*blocking=*/true);
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), stream1_workload, /*blocking=*/true);
    for (uint32_t i = 0; i < kLaunchesPerStream; ++i) {
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), stream0_workload, /*blocking=*/false);
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), stream1_workload, /*blocking=*/false);
    }
    distributed::Finish(mesh_device->mesh_command_queue());

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    EXPECT_EQ(capability->loss.descriptor_full, 0u);

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerConcurrentDevicePath, AllStreamsPreservePublicationOrderUnderStress) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kFirstRuntimeId = 0x7800;
    constexpr uint32_t kRounds = 4;
    constexpr auto kObserverTimeout = std::chrono::seconds(3);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    auto* device = mesh_device->get_devices().front();
    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }
    if (mesh_device->compute_with_storage_grid_size().x < REALTIME_PROFILER_MAX_STREAMS) {
        mesh_device->close();
        GTEST_SKIP() << "Eight independent worker cores are required";
    }

    std::vector<CoreRangeSet> stream_cores;
    std::vector<SubDevice> sub_devices;
    std::vector<distributed::MeshWorkload> workloads;
    stream_cores.reserve(REALTIME_PROFILER_MAX_STREAMS);
    sub_devices.reserve(REALTIME_PROFILER_MAX_STREAMS);
    workloads.reserve(REALTIME_PROFILER_MAX_STREAMS);
    for (uint32_t stream = 0; stream < REALTIME_PROFILER_MAX_STREAMS; ++stream) {
        stream_cores.emplace_back(CoreRange(CoreCoord{stream, 0}, CoreCoord{stream, 0}));
        sub_devices.emplace_back(std::array{stream_cores.back()});
    }
    const auto sub_device_manager = mesh_device->create_sub_device_manager(sub_devices, 3200);
    mesh_device->load_sub_device_manager(sub_device_manager);
    for (uint32_t stream = 0; stream < REALTIME_PROFILER_MAX_STREAMS; ++stream) {
        workloads.emplace_back(make_profiled_data_movement_workload(
            mesh_device, kFirstRuntimeId + stream, stream_cores[stream], 1'000'000));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workloads.back(), /*blocking=*/true);
    }

    const auto l1_view = realtime_profiler_dispatch_l1_view(mesh_device);
    ASSERT_GE(
        wait_for_completed_records(l1_view, REALTIME_PROFILER_MAX_STREAMS, kObserverTimeout),
        REALTIME_PROFILER_MAX_STREAMS);
    for (uint32_t round = 0; round < kRounds; ++round) {
        for (uint32_t stream = 0; stream < REALTIME_PROFILER_MAX_STREAMS; ++stream) {
            distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workloads[stream], /*blocking=*/false);
        }
        distributed::Finish(mesh_device->mesh_command_queue());
        ASSERT_GE(
            wait_for_completed_records(l1_view, (round + 2) * REALTIME_PROFILER_MAX_STREAMS, kObserverTimeout),
            (round + 2) * REALTIME_PROFILER_MAX_STREAMS);
    }

    const uint32_t total_records = (kRounds + 1) * REALTIME_PROFILER_MAX_STREAMS;
    const auto records = read_completed_records(l1_view, total_records);
    ASSERT_EQ(records.size(), total_records * REALTIME_PROFILER_COMPLETED_RECORD_WORDS);
    for (uint32_t record_index = 0; record_index < total_records; ++record_index) {
        const uint32_t* record = &records[record_index * REALTIME_PROFILER_COMPLETED_RECORD_WORDS];
        const uint32_t stream = (record[3] >> 16) & 0xFF;
        ASSERT_LT(stream, REALTIME_PROFILER_MAX_STREAMS);
        EXPECT_EQ(record[2] & 0xFFFF, kFirstRuntimeId + stream);
        EXPECT_EQ(record[3] & 0xFF, REALTIME_PROFILER_RECORD_SCHEMA_VERSION);
        EXPECT_EQ(record[6], record_index + 1);
        EXPECT_GT(
            (static_cast<uint64_t>(record[4]) << 32) | record[5], (static_cast<uint64_t>(record[0]) << 32) | record[1]);
    }

    const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto capability = std::find_if(capabilities.begin(), capabilities.end(), [device](const auto& entry) {
        return entry.chip_id == static_cast<uint32_t>(device->id());
    });
    ASSERT_NE(capability, capabilities.end());
    EXPECT_EQ(capability->loss.descriptor_full, 0u);
    EXPECT_EQ(capability->loss.observer_coalesced, 0u);
    EXPECT_EQ(capability->loss.completed_record, 0u);

    mesh_device->clear_loaded_sub_device_manager();
    mesh_device->remove_sub_device_manager(sub_device_manager);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, CloseDrainsRegisteredCallback) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    std::vector<ProgramRealtimeRecord> records;
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&records](const ProgramRealtimeRecordBatch& batch) {
            records.insert(records.end(), batch.records.begin(), batch.records.end());
        });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});
    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        enqueue_sanity_program(mesh_device, i + 1, all_cores);
    }

    mesh_device->quiesce_devices();
    EXPECT_TRUE(mesh_device->close());

    UnregisterProgramRealtimeProfilerCallback(handle);

    std::set<uint32_t> observed_runtime_ids;
    for (const auto& rec : records) {
        if (rec.runtime_id >= 1 && rec.runtime_id <= kNumPrograms) {
            observed_runtime_ids.insert(rec.runtime_id);
        }
    }
    EXPECT_EQ(observed_runtime_ids.size(), kNumPrograms)
        << "Mesh close should drain records for callbacks still registered at shutdown";
}

TEST(RealtimeProfilerSanity, ThrowingCallbackIsIsolated) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    uint64_t throwing_invocations = 0;
    std::vector<ProgramRealtimeRecord> records;
    ProgramRealtimeProfilerCallbackHandle throwing_handle =
        RegisterProgramRealtimeProfilerCallback([&throwing_invocations](const ProgramRealtimeRecordBatch&) {
            ++throwing_invocations;
            throw std::runtime_error("intentional callback failure");
        });
    ProgramRealtimeProfilerCallbackHandle good_handle =
        RegisterProgramRealtimeProfilerCallback([&records](const ProgramRealtimeRecordBatch& batch) {
            records.insert(records.end(), batch.records.begin(), batch.records.end());
        });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});
    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        enqueue_sanity_program(mesh_device, i + 1, all_cores);
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    UnregisterProgramRealtimeProfilerCallback(throwing_handle);
    UnregisterProgramRealtimeProfilerCallback(good_handle);

    std::set<uint32_t> observed_runtime_ids;
    for (const auto& rec : records) {
        if (rec.runtime_id >= 1 && rec.runtime_id <= kNumPrograms) {
            observed_runtime_ids.insert(rec.runtime_id);
        }
    }
    EXPECT_GT(throwing_invocations, 0u) << "throwing callback should have been invoked";
    EXPECT_EQ(observed_runtime_ids.size(), kNumPrograms)
        << "sibling callback must receive every record despite the other callback throwing";
}

TEST(RealtimeProfilerSanity, LastProgramRecordDeliveredOnFinish) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    const uint32_t chip_id = static_cast<uint32_t>(mesh_device->get_devices().front()->id());

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    test_utils::RealtimeProfilerRecordCollector collector;
    ProgramRealtimeProfilerCallbackHandle handle = RegisterProgramRealtimeProfilerCallback(
        [&collector](const ProgramRealtimeRecordBatch& batch) { collector.consume(batch); });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});

    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        enqueue_sanity_program(mesh_device, i + 1, all_cores);
    }

    distributed::Finish(mesh_device->mesh_command_queue());
    constexpr uint32_t last_runtime_id = kNumPrograms;
    const auto wait_result = collector.wait_for_runtime_ids({last_runtime_id}, std::chrono::seconds(2));
    UnregisterProgramRealtimeProfilerCallback(handle);

    EXPECT_EQ(wait_result.host_dropped, 0u) << "Host callback loss makes final-record delivery inconclusive";
    EXPECT_TRUE(wait_result.complete)
        << "The final program's RT profiler record (runtime_id=" << last_runtime_id
        << ") was not delivered after device queue completion and a bounded callback wait";

    EXPECT_TRUE(mesh_device->close());

    const auto final_capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
    const auto final_capability =
        std::find_if(final_capabilities.begin(), final_capabilities.end(), [chip_id](const auto& entry) {
            return entry.chip_id == chip_id;
        });
    ASSERT_NE(final_capability, final_capabilities.end());
    EXPECT_FALSE(final_capability->active);
    EXPECT_EQ(final_capability->loss.terminal_descriptor, 0u);
    EXPECT_EQ(final_capability->loss.terminal_record, 0u);
    EXPECT_EQ(final_capability->loss.observer_stop_timeout, 0u);
}

TEST(RealtimeProfilerSanity, TraceReplayIsUnprofiledAndOrdinaryProfilingResumes) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kWarmupRuntimeId = 0x6001;
    constexpr uint32_t kTraceRuntimeId = 0x6002;
    constexpr uint32_t kResumeRuntimeId = 0x6003;
    constexpr size_t kTraceRegionSize = 8 * 1024 * 1024;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, kTraceRegionSize, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        const auto capabilities = GetProgramRealtimeProfilerDeviceCapabilities();
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config; inactive_reason="
                     << (capabilities.empty() ? -1 : static_cast<int>(capabilities.front().inactive_reason));
    }

    test_utils::RealtimeProfilerRecordCollector collector;
    ProgramRealtimeProfilerCallbackHandle handle = RegisterProgramRealtimeProfilerCallback(
        [&collector](const ProgramRealtimeRecordBatch& batch) { collector.consume(batch); });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});

    const std::string kernel_src = make_sanity_kernel_source(kTraceRuntimeId);
    Program program = CreateProgram();
    CreateKernelFromString(
        program,
        kernel_src,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernelFromString(
        program,
        kernel_src,
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernelFromString(program, kernel_src, all_cores, ComputeConfig{});
    program.set_runtime_id(static_cast<uint64_t>(kWarmupRuntimeId));

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    auto& mesh_cq = mesh_device->mesh_command_queue(0);

    // Warm up before capture (capture cannot load binaries) under kWarmupRuntimeId, then switch to
    // kTraceRuntimeId so the trace-baked id is tied only by create_trace_node, the path under test.
    distributed::EnqueueMeshWorkload(mesh_cq, workload, true);
    distributed::EnqueueMeshWorkload(mesh_cq, workload, true);
    const auto warmup_wait = collector.wait_for_runtime_ids({kWarmupRuntimeId}, std::chrono::seconds(10));
    ASSERT_TRUE(warmup_wait.complete);
    ASSERT_EQ(warmup_wait.host_dropped, 0u);
    for (auto& [_, prog] : workload.get_programs()) {
        prog.set_runtime_id(static_cast<uint64_t>(kTraceRuntimeId));
    }

    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), mesh_cq.id());
    distributed::EnqueueMeshWorkload(mesh_cq, workload, false);
    mesh_device->end_mesh_trace(mesh_cq.id(), trace_id);
    // Negative control for trace suppression: inspect the captured command cache itself, so this test fails if
    // update_traced_program_dispatch_commands stops clearing either profiler field even when replay happens to emit
    // no host-visible record for another reason.
    for (auto& [device_range, captured_program] : workload.get_programs()) {
        (void)device_range;
        auto& trace_commands = captured_program.impl().get_trace_cached_program_command_sequences();
        ASSERT_FALSE(trace_commands.empty());
        for (const auto& [command_hash, command_sequence] : trace_commands) {
            (void)command_hash;
            ASSERT_NE(command_sequence.mcast_go_signal_cmd_ptr, nullptr);
            EXPECT_EQ(command_sequence.mcast_go_signal_cmd_ptr->profiler_num_workers, 0u);
            EXPECT_EQ(command_sequence.mcast_go_signal_cmd_ptr->profiler_runtime_id, 0u);
        }
    }
    mesh_device->replay_mesh_trace(mesh_cq.id(), trace_id, true);

    // Trace go commands carry zero profiler fields. An ordinary launch after replay must still
    // carry its runtime ID and prove that trace suppression did not deactivate the profiler.
    enqueue_sanity_program(mesh_device, kResumeRuntimeId, all_cores);

    mesh_device->quiesce_devices();
    const auto wait_result = collector.wait_for_runtime_ids({kResumeRuntimeId}, std::chrono::seconds(10));
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(trace_id);

    const auto records = collector.records();
    uint32_t trace_records = 0;
    std::set<uint32_t> ordinary_runtime_ids;
    for (const auto& rec : records) {
        if (rec.runtime_id == kTraceRuntimeId) {
            ++trace_records;
        }
        if (rec.runtime_id == kWarmupRuntimeId || rec.runtime_id == kResumeRuntimeId) {
            ordinary_runtime_ids.insert(rec.runtime_id);
        }
    }
    EXPECT_TRUE(wait_result.complete);
    EXPECT_EQ(wait_result.host_dropped, 0u);
    EXPECT_EQ(trace_records, 0u) << "Trace replay must not publish profiler descriptors";
    EXPECT_EQ(ordinary_runtime_ids, (std::set<uint32_t>{kWarmupRuntimeId, kResumeRuntimeId}));

    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
