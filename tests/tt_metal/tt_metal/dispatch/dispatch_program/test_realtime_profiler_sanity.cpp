// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Blackhole real-time-profiler sanity checks. Milestone 1 validates capability,
// nonblocking launch identity, and device start timestamps. Correlated completion
// endpoints are deliberately suppressed until the Milestone 2 descriptor protocol.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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
#include "hostdev/realtime_profiler_msgs.h"
#include "impl/context/metal_context.hpp"
#include "impl/dispatch/command_queue_common.hpp"
#include "impl/dispatch/dispatch_mem_map.hpp"
#include "impl/dispatch/dispatch_settings.hpp"
#include "llrt/hal.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"
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
        "[RT profiler M0] baseline_msg={} B message_budget={} B dispatch_end=0x{:x} l1_size=0x{:x} "
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
        EXPECT_EQ(rec.end_timestamp, 0u)
            << "Milestone 1 must not publish the legacy observer's off-by-one endpoint as a correlated duration";
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
