// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sanity checks for the real-time (RT) profiler on Blackhole single-chip
// configurations. Enqueues a handful of compute
// programs back-to-back on all tensix cores, attaches an RT profiler
// callback, and asserts that each program produces a record with a
// plausible start/end timestamp. The goal is to catch coarse regressions
// in the RT profiler pipeline (mailbox layout, D2H socket init, sync
// handshake, kernel source propagation, timestamp extraction) before they
// reach CI's longer-running profiler test suite.
//
// Lives in the dispatch "basic" test library. On configurations where RT
// profiler cannot be enabled (ETH dispatch, non-MMIO chip, kernels
// nullified, IOMMU-off on BH, etc.) the test skips gracefully via
// IsProgramRealtimeProfilerActive().

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <future>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>

#include "tt_metal/distributed/mesh_device_impl.hpp"
#include "tt_metal/distributed/realtime_profiler_manager.hpp"
#include "hostdev/realtime_profiler_msgs.h"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_protocol.hpp"
#include "tt_metal/impl/dispatch/kernels/realtime_profiler_ring_buffer.hpp"

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::FinishAndCollectProgramRealtimeProfiler;
using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeRecord;
using tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

constexpr uint32_t kNumPrograms = 5;
// Generous upper bound: the inlined NOP loop kernels below run ~40K
// unrolled NOPs. Even on slow silicon that stays in the tens-of-microseconds
// range, so 1s is a sanity cap only intended to catch a broken clock /
// mis-decoded timestamp.
constexpr double kMaxDurationNs = 1'000'000'000.0;

// Per-program marker embedded in the kernel source so the source-correlation
// assertion can verify each record carries the correct source.
constexpr const char* kSourceMarkerPrefix = "rt_profiler_marker_";

struct ScopedEnvUnset {
    const char* name;
    ~ScopedEnvUnset() { ::unsetenv(name); }
};

TEST(RealtimeProfilerProtocol, CompletionCounterAndQueueIndicesWrapWithoutAmbiguity) {
    constexpr uint32_t kCounterWidth = 17;
    constexpr uint32_t kCounterMax = (1u << kCounterWidth) - 1;

    EXPECT_EQ(realtime_profiler_completion_target<kCounterWidth>(kCounterMax - 1, 3), 1u);
    EXPECT_TRUE(realtime_profiler_stream_count_ge<kCounterWidth>(0, kCounterMax));
    EXPECT_TRUE(realtime_profiler_stream_count_ge<kCounterWidth>(1, 1));
    EXPECT_TRUE(realtime_profiler_stream_count_ge<kCounterWidth>(1, 0));
    EXPECT_FALSE(realtime_profiler_stream_count_ge<kCounterWidth>(kCounterMax, 0));
    EXPECT_FALSE(realtime_profiler_stream_count_ge<kCounterWidth>(0, 1));
    EXPECT_TRUE(realtime_profiler_generation_after(1, 0));
    EXPECT_FALSE(realtime_profiler_generation_after(0, 1));
    EXPECT_TRUE(realtime_profiler_generation_after(0, UINT32_MAX));

    constexpr uint32_t read_index = UINT32_MAX - 1;
    EXPECT_FALSE(realtime_profiler_queue_full(read_index + 3, read_index, 4));
    EXPECT_TRUE(realtime_profiler_queue_full(read_index + 4, read_index, 4));
    EXPECT_TRUE(realtime_profiler_queue_full(read_index + 5, read_index, 4));
    EXPECT_EQ(realtime_profiler_completion_target<kCounterWidth>(0x20005, 3), 8u);
    EXPECT_FALSE(realtime_profiler_stream_count_ge<kCounterWidth>(0, 1u << (kCounterWidth - 1)));
    EXPECT_TRUE(realtime_profiler_generation_after(0, UINT32_MAX));
    EXPECT_FALSE(realtime_profiler_generation_after(UINT32_MAX, 0));

    const auto& factory =
        MetalContext::instance().hal().get_realtime_profiler_msgs_factory(HalProgrammableCoreType::TENSIX);
    using GeneratedMessage = realtime_profiler_msgs::realtime_profiler_msg_t;
    EXPECT_EQ(factory.size_of<GeneratedMessage>(), sizeof(::realtime_profiler_msg_t));
    EXPECT_EQ(sizeof(::realtime_profiler_msg_t), 5536u);
    EXPECT_EQ(
        factory.offset_of<GeneratedMessage>(GeneratedMessage::Field::kernel_start_b),
        offsetof(::realtime_profiler_msg_t, kernel_start_b));
}

TEST(RealtimeProfilerSanity, DisabledProfilerStillStopsDispatchObserver) {
    constexpr int kDeviceId = 0;
    ASSERT_EQ(::setenv("TT_METAL_DISABLE_REALTIME_PROFILER", "1", /*overwrite=*/1), 0);
    ScopedEnvUnset restore_env{"TT_METAL_DISABLE_REALTIME_PROFILER"};

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    EXPECT_FALSE(IsProgramRealtimeProfilerActive());
    const auto inactive =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::milliseconds(20));
    EXPECT_TRUE(inactive.profiler_inactive);
    EXPECT_FALSE(inactive.protocol_error);
    EXPECT_TRUE(inactive.lossy());
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

std::string make_concurrent_kernel_source(uint32_t outer_iterations) {
    return "#include <cstdint>\n"
           "void kernel_main() {\n"
           "    for (volatile uint32_t i = 0; i < " +
           std::to_string(outer_iterations) +
           "; ++i) {\n"
           "#pragma GCC unroll 200\n"
           "        for (uint32_t j = 0; j < 200; ++j) { asm volatile(\"nop\"); }\n"
           "    }\n"
           "}\n";
}

void enqueue_concurrent_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreCoord& core,
    uint32_t runtime_id,
    uint32_t outer_iterations) {
    Program program = CreateProgram();
    const std::string source = make_concurrent_kernel_source(outer_iterations);
    CreateKernelFromString(
        program,
        source,
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernelFromString(
        program,
        source,
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernelFromString(program, source, core, ComputeConfig{});
    program.set_runtime_id(runtime_id);

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
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
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
    // Give the RT profiler receiver thread a moment to drain the last
    // socket pages before we unregister. 500ms mirrors the programming
    // example at test_realtime_profiler_csv.cpp and has proven sufficient
    // for small workloads on WH/BH single-chip.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    auto* rt_profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt_profiler, nullptr);
    const auto device_losses = rt_profiler->device_loss_counts();
    const auto transport_drops = rt_profiler->transport_drop_count();

    UnregisterProgramRealtimeProfilerCallback(handle);

    std::string observed_ids;
    for (const auto& record : records) {
        observed_ids += fmt::format("{} ", record.runtime_id);
    }
    ASSERT_GE(records.size(), kNumPrograms)
        << "Expected at least " << kNumPrograms << " RT profiler records (one per program), got " << records.size()
        << "; start drops=" << device_losses.start_descriptor
        << ", observer drops=" << device_losses.completion_observer
        << ", record drops=" << device_losses.completed_record << ", transport drops=" << transport_drops
        << "; observed runtime IDs: " << observed_ids;
    EXPECT_EQ(dropped, 0u);

    for (const auto& rec : records) {
        EXPECT_GT(rec.end_timestamp, rec.start_timestamp)
            << "RT record end_timestamp must be strictly greater than start_timestamp (runtime_id=" << rec.runtime_id
            << ", chip=" << rec.chip_id << ")";
        EXPECT_GT(rec.frequency, 0.0) << "RT record frequency must be positive (runtime_id=" << rec.runtime_id
                                      << ", chip=" << rec.chip_id << ")";

        if (rec.frequency > 0.0 && rec.end_timestamp > rec.start_timestamp) {
            uint64_t duration_cycles = rec.end_timestamp - rec.start_timestamp;
            double duration_ns = static_cast<double>(duration_cycles) / rec.frequency;
            EXPECT_LT(duration_ns, kMaxDurationNs)
                << "RT record duration is implausibly large (runtime_id=" << rec.runtime_id << ", chip=" << rec.chip_id
                << ", duration_ns=" << duration_ns << ")";
        }
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

TEST(RealtimeProfilerSanity, ConcurrentPartitionedSubDevicesUseIndependentCompletionTargets) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kSlowRuntimeId = 0x7101;
    constexpr uint32_t kFastRuntimeId = 0x7102;
    constexpr uint32_t kThirdRuntimeId = 0x7103;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x < 3) {
        mesh_device->close();
        GTEST_SKIP() << "Concurrent sub-device profiler test requires three Tensix cores in one row";
    }
    const CoreCoord slow_core{0, 0};
    const CoreCoord fast_core{1, 0};
    const CoreCoord third_core{2, 0};
    SubDevice slow_sub_device(std::array{CoreRangeSet(CoreRange(slow_core, slow_core))});
    SubDevice fast_sub_device(std::array{CoreRangeSet(CoreRange(fast_core, fast_core))});
    SubDevice third_sub_device(std::array{CoreRangeSet(CoreRange(third_core, third_core))});
    auto manager = mesh_device->create_sub_device_manager({slow_sub_device, fast_sub_device, third_sub_device}, 3200);
    mesh_device->load_sub_device_manager(manager);

    // Compile and cache both kernel shapes before the measured launches. Otherwise
    // host-side JIT work for the second program can outlast the first device program,
    // preventing the two asynchronous launches from overlapping at all.
    enqueue_concurrent_program(mesh_device, slow_core, /*runtime_id=*/0, 200000);
    enqueue_concurrent_program(mesh_device, fast_core, /*runtime_id=*/0, 20);
    enqueue_concurrent_program(mesh_device, third_core, /*runtime_id=*/0, 20);
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<ProgramRealtimeRecord> records;
    uint64_t callback_drops = 0;
    const auto handle =
        RegisterProgramRealtimeProfilerCallback([&records, &callback_drops](const ProgramRealtimeRecordBatch& batch) {
            callback_drops += batch.dropped;
            records.insert(records.end(), batch.records.begin(), batch.records.end());
        });

    // Launch the long stream first, then a much shorter program on an independent
    // stream. Correct profiler completion targets produce overlapping intervals
    // with the second stream completing first.
    enqueue_concurrent_program(mesh_device, slow_core, kSlowRuntimeId, 200000);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});
    enqueue_concurrent_program(mesh_device, fast_core, kFastRuntimeId, 20);
    mesh_device->set_sub_device_stall_group({{SubDeviceId{2}}});
    enqueue_concurrent_program(mesh_device, third_core, kThirdRuntimeId, 20);
    mesh_device->reset_sub_device_stall_group();
    distributed::Finish(mesh_device->mesh_command_queue());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    auto* rt_profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt_profiler, nullptr);
    const auto device_losses = rt_profiler->device_loss_counts();
    EXPECT_EQ(device_losses.total(), 0u);
    EXPECT_EQ(rt_profiler->transport_drop_count(), 0u);
    UnregisterProgramRealtimeProfilerCallback(handle);

    const ProgramRealtimeRecord* slow = nullptr;
    const ProgramRealtimeRecord* fast = nullptr;
    const ProgramRealtimeRecord* third = nullptr;
    std::string observed_runtime_ids;
    for (const auto& record : records) {
        observed_runtime_ids +=
            fmt::format("{}:[{},{}] ", record.runtime_id, record.start_timestamp, record.end_timestamp);
        if (record.runtime_id == kSlowRuntimeId) {
            slow = &record;
        } else if (record.runtime_id == kFastRuntimeId) {
            fast = &record;
        } else if (record.runtime_id == kThirdRuntimeId) {
            third = &record;
        }
    }
    ASSERT_NE(slow, nullptr) << "Missing interval from sub-device 0; observed runtime IDs: " << observed_runtime_ids;
    ASSERT_NE(fast, nullptr) << "Missing interval from sub-device 1";
    ASSERT_NE(third, nullptr) << "Missing interval from sub-device 2";
    EXPECT_EQ(callback_drops, 0u);
    EXPECT_LT(slow->start_timestamp, fast->end_timestamp);
    EXPECT_LT(fast->start_timestamp, slow->end_timestamp);
    EXPECT_LT(fast->end_timestamp, slow->end_timestamp)
        << "The short program on the second stream should complete before the long first-stream program"
        << " (slow=[" << slow->start_timestamp << ", " << slow->end_timestamp << "], fast=[" << fast->start_timestamp
        << ", " << fast->end_timestamp << "])";
    EXPECT_LT(third->start_timestamp, slow->end_timestamp);
    EXPECT_LT(third->end_timestamp, slow->end_timestamp)
        << "The third sub-device interval validates go-command fields that share the in-place staging word";

    mesh_device->clear_loaded_sub_device_manager();
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, DeviceIntervalsProveOverlapAndSerialization) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kOverlapSlowId = 0x7801;
    constexpr uint32_t kOverlapFastId = 0x7802;
    constexpr uint32_t kSerialFirstId = 0x7803;
    constexpr uint32_t kSerialSecondId = 0x7804;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x < 2) {
        mesh_device->close();
        GTEST_SKIP() << "End-to-end interval test requires two Tensix cores in one row";
    }
    const std::array cores{CoreCoord{0, 0}, CoreCoord{1, 0}};
    std::vector<SubDevice> sub_devices;
    sub_devices.reserve(cores.size());
    for (const auto& core : cores) {
        sub_devices.emplace_back(std::array{CoreRangeSet(CoreRange(core, core))});
    }
    auto manager = mesh_device->create_sub_device_manager(sub_devices, 3200);
    mesh_device->load_sub_device_manager(manager);

    // Compile both kernel shapes outside the measured phases. Runtime IDs are
    // host metadata, so the two serialized launches below reuse one identical
    // program binary while retaining distinct invocation identities.
    enqueue_concurrent_program(mesh_device, cores[0], /*runtime_id=*/0, 200000);
    enqueue_concurrent_program(mesh_device, cores[1], /*runtime_id=*/0, 20);
    distributed::Finish(mesh_device->mesh_command_queue());

    std::mutex records_mu;
    std::condition_variable records_cv;
    std::vector<ProgramRealtimeRecord> records;
    uint64_t callback_drops = 0;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        {
            std::lock_guard<std::mutex> lock(records_mu);
            callback_drops += batch.dropped;
            records.insert(records.end(), batch.records.begin(), batch.records.end());
        }
        records_cv.notify_all();
    });

    enqueue_concurrent_program(mesh_device, cores[0], kOverlapSlowId, 200000);
    // Restrict implicit host-side stalls to stream 1 so dispatch can launch
    // the second program while stream 0 is still running.
    mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});
    enqueue_concurrent_program(mesh_device, cores[1], kOverlapFastId, 20);
    mesh_device->reset_sub_device_stall_group();
    const auto overlap_collection =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));

    const std::array first_stream{SubDeviceId{0}};
    enqueue_concurrent_program(mesh_device, cores[0], kSerialFirstId, 20);
    const auto serial_first_collection = FinishAndCollectProgramRealtimeProfiler(
        mesh_device->mesh_command_queue(), std::chrono::seconds(5), first_stream);
    const std::array second_stream{SubDeviceId{1}};
    enqueue_concurrent_program(mesh_device, cores[1], kSerialSecondId, 20);
    const auto serial_second_collection = FinishAndCollectProgramRealtimeProfiler(
        mesh_device->mesh_command_queue(), std::chrono::seconds(5), second_stream);

    auto is_target = [](uint32_t runtime_id) { return runtime_id >= kOverlapSlowId && runtime_id <= kSerialSecondId; };
    bool received_all = false;
    {
        std::unique_lock<std::mutex> lock(records_mu);
        received_all = records_cv.wait_for(lock, std::chrono::seconds(5), [&] {
            return std::count_if(records.begin(), records.end(), [&](const auto& record) {
                       return is_target(record.runtime_id);
                   }) >= 4;
        });
    }
    UnregisterProgramRealtimeProfilerCallback(handle);
    if (!received_all) {
        ADD_FAILURE() << "Timed out waiting for the four target device intervals";
        mesh_device->clear_loaded_sub_device_manager();
        EXPECT_TRUE(mesh_device->close());
        return;
    }

    EXPECT_TRUE(overlap_collection.complete());
    EXPECT_FALSE(overlap_collection.lossy());
    EXPECT_TRUE(serial_first_collection.complete());
    EXPECT_FALSE(serial_first_collection.lossy());
    EXPECT_TRUE(serial_second_collection.complete());
    EXPECT_FALSE(serial_second_collection.lossy());

    std::array<const ProgramRealtimeRecord*, 4> target{};
    std::set<uint32_t> sequences;
    {
        std::lock_guard<std::mutex> lock(records_mu);
        for (const auto& record : records) {
            if (!is_target(record.runtime_id)) {
                continue;
            }
            const auto index = record.runtime_id - kOverlapSlowId;
            if (index >= target.size()) {
                ADD_FAILURE() << "Out-of-range target runtime ID " << record.runtime_id;
                continue;
            }
            if (target[index] != nullptr) {
                ADD_FAILURE() << "Duplicate interval for runtime ID " << record.runtime_id;
                continue;
            }
            target[index] = &record;
            EXPECT_EQ(record.command_queue_id, 0u);
            EXPECT_GT(record.sequence, 0u);
            EXPECT_TRUE(sequences.insert(record.sequence).second);
            EXPECT_GT(record.end_timestamp, record.start_timestamp);
        }
    }
    bool have_every_target = true;
    for (uint32_t i = 0; i < target.size(); ++i) {
        if (target[i] == nullptr) {
            ADD_FAILURE() << "Missing interval for runtime ID " << kOverlapSlowId + i;
            have_every_target = false;
        }
    }
    if (!have_every_target) {
        mesh_device->clear_loaded_sub_device_manager();
        EXPECT_TRUE(mesh_device->close());
        return;
    }
    EXPECT_EQ(callback_drops, 0u);
    EXPECT_EQ(target[0]->dispatch_stream, 0u);
    EXPECT_EQ(target[1]->dispatch_stream, 1u);
    EXPECT_EQ(target[2]->dispatch_stream, 0u);
    EXPECT_EQ(target[3]->dispatch_stream, 1u);

    // These are raw device ticks in one device clock domain. No host timestamp
    // participates in either the overlap or serialization decision.
    EXPECT_LT(target[0]->start_timestamp, target[1]->end_timestamp);
    EXPECT_LT(target[1]->start_timestamp, target[0]->end_timestamp);
    EXPECT_LT(target[1]->end_timestamp, target[0]->end_timestamp);
    EXPECT_LT(target[2]->end_timestamp, target[3]->start_timestamp);

    auto* profiler = mesh_device->impl().get_realtime_profiler();
    if (profiler == nullptr) {
        ADD_FAILURE() << "Missing active real-time profiler manager";
        mesh_device->clear_loaded_sub_device_manager();
        EXPECT_TRUE(mesh_device->close());
        return;
    }
    const auto losses = profiler->device_loss_counts();
    EXPECT_EQ(losses.total(), 0u);
    EXPECT_EQ(profiler->transport_drop_count(), 0u);
    log_info(
        tt::LogTest,
        "[RT profiler qualification intervals] overlap_slow=[{},{}] overlap_fast=[{},{}] "
        "serial_first=[{},{}] serial_second=[{},{}] source_loss={} transport_loss={}",
        target[0]->start_timestamp,
        target[0]->end_timestamp,
        target[1]->start_timestamp,
        target[1]->end_timestamp,
        target[2]->start_timestamp,
        target[2]->end_timestamp,
        target[3]->start_timestamp,
        target[3]->end_timestamp,
        losses.total(),
        profiler->transport_drop_count());

    mesh_device->clear_loaded_sub_device_manager();
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, QualificationHookMeasuresObserverDeviceCycles) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kProgramsPerStream = 64;
    if (std::getenv("TT_RT_PROFILER_QUALIFICATION_HOOK") == nullptr) {
        GTEST_SKIP() << "Set TT_RT_PROFILER_QUALIFICATION_HOOK before process startup to run this qualification";
    }

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (mesh_device->impl().get_device(0)->arch() != tt::ARCH::BLACKHOLE) {
        mesh_device->close();
        GTEST_SKIP() << "Observer-cycle qualification is Blackhole-only";
    }
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this Blackhole dispatch config";
    }

    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x < realtime_profiler_msgs::REALTIME_PROFILER_MAX_STREAMS) {
        mesh_device->close();
        GTEST_SKIP() << "Observer-cycle qualification requires eight Tensix cores in one row";
    }
    const std::array cores{
        CoreCoord{0, 0},
        CoreCoord{1, 0},
        CoreCoord{2, 0},
        CoreCoord{3, 0},
        CoreCoord{4, 0},
        CoreCoord{5, 0},
        CoreCoord{6, 0},
        CoreCoord{7, 0}};
    std::vector<SubDevice> sub_devices;
    sub_devices.reserve(cores.size());
    for (const auto& core : cores) {
        sub_devices.emplace_back(std::array{CoreRangeSet(CoreRange(core, core))});
    }
    auto manager = mesh_device->create_sub_device_manager(sub_devices, 3200);
    mesh_device->load_sub_device_manager(manager);

    enqueue_concurrent_program(mesh_device, cores[0], /*runtime_id=*/0, 20);
    enqueue_concurrent_program(mesh_device, cores[1], /*runtime_id=*/0, 20);
    const auto warmup =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    if (!warmup.complete() || warmup.lossy()) {
        ADD_FAILURE() << "Qualification warmup collection was incomplete or lossy";
        mesh_device->clear_loaded_sub_device_manager();
        EXPECT_TRUE(mesh_device->close());
        return;
    }

    auto* profiler = mesh_device->impl().get_realtime_profiler();
    if (profiler == nullptr) {
        ADD_FAILURE() << "Missing active real-time profiler manager";
        mesh_device->clear_loaded_sub_device_manager();
        EXPECT_TRUE(mesh_device->close());
        return;
    }
    const auto before = profiler->qualification_counts_for_testing();
    for (uint32_t i = 0; i < kProgramsPerStream; ++i) {
        enqueue_concurrent_program(mesh_device, cores[0], 0x7900 + 2 * i, 20);
        mesh_device->set_sub_device_stall_group({{SubDeviceId{1}}});
        enqueue_concurrent_program(mesh_device, cores[1], 0x7901 + 2 * i, 20);
        mesh_device->reset_sub_device_stall_group();
    }
    const auto collection =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(10));
    // Counter stores precede the record's watermark, while the maximum-scan
    // store occurs at the end of the enclosing observer pass. Let that final
    // pass retire before reading the test-only scratch words.
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto after = profiler->qualification_counts_for_testing();

    const uint64_t handler_cycles = after.record_handler_cycles - before.record_handler_cycles;
    const uint32_t handler_count = after.record_handler_count - before.record_handler_count;
    const double cycles_per_record =
        handler_count == 0 ? 0.0 : static_cast<double>(handler_cycles) / static_cast<double>(handler_count);
    const double max_scan_ns = after.minimum_frequency > 0.0
                                   ? static_cast<double>(after.max_observer_scan_cycles) / after.minimum_frequency
                                   : 0.0;
    const auto losses = profiler->device_loss_counts();
    const uint32_t transport_drops = profiler->transport_drop_count();
    log_info(
        tt::LogTest,
        "[RT profiler qualification observer] watermark_receiver_delta={} handler_count={} handler_cycles={} "
        "cycles_per_record={:.2f} max_scan_cycles={} frequency={:.6f} cycles/ns max_scan_ns={:.2f} "
        "source_loss={} transport_loss={}",
        collection.record_count,
        handler_count,
        handler_cycles,
        cycles_per_record,
        after.max_observer_scan_cycles,
        after.minimum_frequency,
        max_scan_ns,
        losses.total(),
        transport_drops);

    EXPECT_TRUE(collection.complete());
    EXPECT_FALSE(collection.lossy());
    // collection.record_count is the host-receiver delta after the watermark
    // was registered; records already drained concurrently may precede that
    // baseline. The device-side handler delta is the authoritative workload
    // count for this qualification hook.
    EXPECT_EQ(handler_count, 2u * kProgramsPerStream);
    EXPECT_GT(handler_cycles, 0u);
    EXPECT_GT(after.max_observer_scan_cycles, 0u);
    EXPECT_GT(after.minimum_frequency, 0.0);
    EXPECT_EQ(losses.total(), 0u);
    EXPECT_EQ(transport_drops, 0u);

    mesh_device->clear_loaded_sub_device_manager();
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, ExactWatermarkCollectsEverySelectedStream) {
    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    if (grid.x < 3) {
        mesh_device->close();
        GTEST_SKIP() << "Watermark test requires three Tensix cores in one row";
    }
    const std::array cores{CoreCoord{0, 0}, CoreCoord{1, 0}, CoreCoord{2, 0}};
    std::vector<SubDevice> sub_devices;
    sub_devices.reserve(cores.size());
    for (const auto& core : cores) {
        sub_devices.emplace_back(std::array{CoreRangeSet(CoreRange(core, core))});
    }
    auto manager = mesh_device->create_sub_device_manager(sub_devices, 3200);
    mesh_device->load_sub_device_manager(manager);

    std::mutex records_mu;
    std::vector<ProgramRealtimeRecord> records;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        std::lock_guard<std::mutex> lock(records_mu);
        records.insert(records.end(), batch.records.begin(), batch.records.end());
    });

    for (uint32_t stream = 0; stream < cores.size(); ++stream) {
        mesh_device->set_sub_device_stall_group({{SubDeviceId{stream}}});
        enqueue_concurrent_program(mesh_device, cores[stream], 0x7301 + stream, 20 + stream * 10);
    }
    mesh_device->reset_sub_device_stall_group();

    const auto result =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    EXPECT_TRUE(result.complete());
    EXPECT_FALSE(result.timed_out);
    EXPECT_FALSE(result.protocol_error);
    EXPECT_EQ(result.source_dropped, 0u);
    EXPECT_EQ(result.transport_dropped, 0u);
    ASSERT_EQ(result.devices.size(), 1u);
    EXPECT_EQ(result.devices[0].expected_stream_mask, 0x7u);
    EXPECT_EQ(result.devices[0].observed_stream_mask, 0x7u);

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    UnregisterProgramRealtimeProfilerCallback(handle);
    {
        std::lock_guard<std::mutex> lock(records_mu);
        std::set<uint32_t> observed_streams;
        for (const auto& record : records) {
            if (record.runtime_id >= 0x7301 && record.runtime_id <= 0x7303) {
                EXPECT_EQ(record.schema_version, realtime_profiler_msgs::REALTIME_PROFILER_RECORD_SCHEMA_VERSION);
                EXPECT_EQ(record.record_type, realtime_profiler_msgs::REALTIME_PROFILER_RECORD_TYPE_INTERVAL);
                EXPECT_GT(record.sequence, 0u);
                observed_streams.insert(record.dispatch_stream);
            }
        }
        EXPECT_EQ(observed_streams, (std::set<uint32_t>{0, 1, 2}));
    }

    mesh_device->clear_loaded_sub_device_manager();
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, CollectionTimeoutDoesNotUseRingEmptinessAndRecovers) {
    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    auto* profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(profiler, nullptr);
    profiler->set_receiver_loop_paused_for_testing(true);
    const auto timed_out =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::milliseconds(20));
    EXPECT_TRUE(timed_out.timed_out);
    EXPECT_TRUE(timed_out.lossy());
    EXPECT_FALSE(timed_out.complete());
    ASSERT_FALSE(timed_out.devices.empty());
    EXPECT_NE(timed_out.devices[0].expected_stream_mask, 0u);
    EXPECT_EQ(timed_out.devices[0].observed_stream_mask, 0u);

    profiler->set_receiver_loop_paused_for_testing(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto recovered =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    EXPECT_TRUE(recovered.complete());
    EXPECT_FALSE(recovered.timed_out);
    EXPECT_FALSE(recovered.protocol_error);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, WatermarkIdWrapSkipsReservedZero) {
    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    auto* profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(profiler, nullptr);
    profiler->set_next_watermark_id_for_testing(UINT32_MAX);
    const auto before_wrap =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    const auto after_wrap =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    EXPECT_TRUE(before_wrap.complete());
    EXPECT_EQ(before_wrap.requested_watermark, UINT32_MAX);
    EXPECT_TRUE(after_wrap.complete());
    EXPECT_EQ(after_wrap.requested_watermark, 1u);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, UnexpectedWatermarkReturnsPromptProtocolError) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kWatermarkId = 0x7501;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    auto* profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(profiler, nullptr);
    profiler->set_receiver_loop_paused_for_testing(true);
    profiler->set_next_watermark_id_for_testing(kWatermarkId);
    auto collection = std::async(std::launch::async, [&] {
        return FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    });

    bool injected = false;
    for (uint32_t attempt = 0; attempt < 100 && !injected; ++attempt) {
        injected = profiler->inject_unexpected_watermark_for_testing(kWatermarkId, /*stream=*/31);
        if (!injected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    EXPECT_TRUE(injected);
    const auto prompt_status = collection.wait_for(std::chrono::milliseconds(100));
    profiler->set_receiver_loop_paused_for_testing(false);
    const auto result = collection.get();
    EXPECT_EQ(prompt_status, std::future_status::ready);
    EXPECT_TRUE(result.protocol_error);
    EXPECT_FALSE(result.complete());
    EXPECT_FALSE(result.timed_out);
    EXPECT_TRUE(result.lossy());
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, BackToBackExactWatermarksDoNotStrandReadySlot) {
    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    constexpr uint32_t kIterations = 64;
    for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        const auto result =
            FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(1));
        ASSERT_TRUE(result.complete()) << "watermark stranded at iteration " << iteration;
        ASSERT_FALSE(result.timed_out) << "watermark timed out at iteration " << iteration;
    }
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, ReservedRingDropsIntervalsAndRetainsFullRingWatermark) {
    constexpr int kDeviceId = 0;
    ASSERT_EQ(::setenv("TT_RT_PROFILER_RING_TEST_HOOK", "1", /*overwrite=*/1), 0);
    ScopedEnvUnset restore_env{"TT_RT_PROFILER_RING_TEST_HOOK"};
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    const CoreCoord core{0, 0};
    SubDevice sub_device(std::array{CoreRangeSet(CoreRange(core, core))});
    auto manager = mesh_device->create_sub_device_manager({sub_device}, 3200);
    mesh_device->load_sub_device_manager(manager);

    // Warm compilation before the injection window, then hold NCRISC only at
    // the pathological interval-full threshold. The production image does not
    // compile this hook.
    enqueue_concurrent_program(mesh_device, core, /*runtime_id=*/0, 20);
    distributed::Finish(mesh_device->mesh_command_queue());
    auto* profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(profiler, nullptr);
    const uint32_t initial_transport_drops = profiler->transport_drop_count();
    profiler->prime_reserved_ring_for_testing(RT_PROFILER_RING_CAPACITY - 1);
    EXPECT_EQ(profiler->reserved_ring_occupancy_for_testing(), RT_PROFILER_RING_CAPACITY - 1);

    // The reserved control slot must make interval pressure lossy instead of
    // blocking dispatch.
    enqueue_concurrent_program(mesh_device, core, /*runtime_id=*/0x7401, 20);
    distributed::Finish(mesh_device->mesh_command_queue());
    for (uint32_t attempt = 0; attempt < 100 && profiler->transport_drop_count() == initial_transport_drops;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GT(profiler->transport_drop_count(), initial_transport_drops);

    // The first watermark occupies the reserved slot. The second reaches the
    // dispatch mailbox while the ring is completely full and must remain
    // pending, without blocking dispatch_s, until NCRISC resumes.
    auto first = std::async(std::launch::async, [&] {
        return FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    });
    for (uint32_t attempt = 0;
         attempt < 100 && profiler->reserved_ring_occupancy_for_testing() != RT_PROFILER_RING_CAPACITY;
         ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_EQ(profiler->reserved_ring_occupancy_for_testing(), RT_PROFILER_RING_CAPACITY);

    auto second = std::async(std::launch::async, [&] {
        return FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    });
    for (uint32_t attempt = 0; attempt < 100 && !profiler->dispatch_mailbox_pending_for_testing(); ++attempt) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_TRUE(profiler->dispatch_mailbox_pending_for_testing());
    EXPECT_EQ(first.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);
    EXPECT_EQ(second.wait_for(std::chrono::milliseconds(0)), std::future_status::timeout);

    profiler->resume_reserved_ring_consumer_for_testing();
    const auto first_result = first.get();
    const auto second_result = second.get();
    EXPECT_TRUE(first_result.complete());
    EXPECT_TRUE(second_result.complete());
    EXPECT_GT(first_result.transport_dropped, 0u);

    mesh_device->clear_loaded_sub_device_manager();
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, FullDeviceQueuesCountDropsWithoutStallingDispatch) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kRuntimeId = 0x7201;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    const CoreCoord core{0, 0};
    SubDevice sub_device(std::array{CoreRangeSet(CoreRange(core, core))});
    auto manager = mesh_device->create_sub_device_manager({sub_device}, 3200);
    mesh_device->load_sub_device_manager(manager);

    // Warm the program cache before injecting queue state so compilation is
    // outside the fault-injection window.
    enqueue_concurrent_program(mesh_device, core, /*runtime_id=*/0, 20);
    distributed::Finish(mesh_device->mesh_command_queue());

    auto* rt_profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt_profiler, nullptr);
    const auto initial_losses = rt_profiler->device_loss_counts();

    // A target well ahead of the fresh stream counter keeps the synthetic
    // descriptors queued while dispatch_s attempts to publish this launch.
    rt_profiler->prime_start_descriptor_queue_full_for_testing(/*stream_index=*/0, /*completion_target=*/4096);
    enqueue_concurrent_program(mesh_device, core, kRuntimeId, 20);
    distributed::Finish(mesh_device->mesh_command_queue());
    const auto start_full_losses = rt_profiler->device_loss_counts();
    EXPECT_GT(start_full_losses.start_descriptor, initial_losses.start_descriptor);
    EXPECT_GT(start_full_losses.stuck_descriptor_head, initial_losses.stuck_descriptor_head);
    rt_profiler->clear_start_descriptor_queue_for_testing(/*stream_index=*/0);

    // Hold the completed-record queue in a synthetic over-capacity state. The
    // observer must account a ready interval instead of waiting for space.
    rt_profiler->prime_completed_record_queue_full_for_testing();
    rt_profiler->prime_start_descriptor_queue_full_for_testing(/*stream_index=*/0, /*completion_target=*/0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto record_full_losses = rt_profiler->device_loss_counts();
    EXPECT_GT(record_full_losses.completed_record, start_full_losses.completed_record)
        << "losses after injected full record queue: start=" << record_full_losses.start_descriptor
        << ", unsupported=" << record_full_losses.unsupported_launch
        << ", reset=" << record_full_losses.reset_descriptor << ", observer=" << record_full_losses.completion_observer
        << ", record=" << record_full_losses.completed_record;
    rt_profiler->clear_start_descriptor_queue_for_testing(/*stream_index=*/0);
    rt_profiler->clear_completed_record_queue_for_testing();

    mesh_device->clear_loaded_sub_device_manager();
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, CompletionObserverAccountsMultiReadyAndResetDescriptors) {
    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    auto* rt_profiler = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt_profiler, nullptr);
    const auto initial_losses = rt_profiler->device_loss_counts();

    // On a fresh device stream 0 is at count zero, so all four synthetic
    // descriptors become ready in one scan. The observer keeps the newest
    // interval and accounts the other three.
    rt_profiler->prime_start_descriptor_queue_full_for_testing(/*stream_index=*/0, /*completion_target=*/0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto multi_ready_losses = rt_profiler->device_loss_counts();
    EXPECT_GE(
        multi_ready_losses.completion_observer,
        initial_losses.completion_observer + realtime_profiler_msgs::REALTIME_PROFILER_START_QUEUE_CAPACITY - 1);
    rt_profiler->clear_start_descriptor_queue_for_testing(/*stream_index=*/0);

    // Publish old-generation descriptors with an unmet target, then advance
    // the reset epoch. The observer must discard all stale entries explicitly.
    rt_profiler->prime_start_descriptor_queue_full_for_testing(/*stream_index=*/0, /*completion_target=*/4096);
    rt_profiler->advance_stream_reset_generation_for_testing(/*stream_index=*/0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto reset_losses = rt_profiler->device_loss_counts();
    EXPECT_GE(
        reset_losses.reset_descriptor,
        multi_ready_losses.reset_descriptor + realtime_profiler_msgs::REALTIME_PROFILER_START_QUEUE_CAPACITY);
    rt_profiler->clear_start_descriptor_queue_for_testing(/*stream_index=*/0);

    const auto collection =
        FinishAndCollectProgramRealtimeProfiler(mesh_device->mesh_command_queue(), std::chrono::seconds(5));
    EXPECT_TRUE(collection.complete());
    EXPECT_GT(collection.source_dropped, 0u);
    EXPECT_EQ(
        collection.source_dropped,
        collection.descriptor_dropped + collection.observer_dropped + collection.record_dropped);

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, CloseDrainsRegisteredCallback) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
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

    std::mutex records_mu;
    std::vector<ProgramRealtimeRecord> records;
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&records_mu, &records](const ProgramRealtimeRecordBatch& batch) {
            std::lock_guard<std::mutex> lk(records_mu);
            records.insert(records.end(), batch.records.begin(), batch.records.end());
        });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});

    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        enqueue_sanity_program(mesh_device, i + 1, all_cores);
    }

    distributed::Finish(mesh_device->mesh_command_queue());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    UnregisterProgramRealtimeProfilerCallback(handle);

    constexpr uint32_t last_runtime_id = kNumPrograms;
    bool last_record_seen = false;
    {
        std::lock_guard<std::mutex> lk(records_mu);
        for (const auto& rec : records) {
            if (rec.runtime_id == last_runtime_id) {
                last_record_seen = true;
                break;
            }
        }
    }

    EXPECT_TRUE(last_record_seen) << "The final program's RT profiler record (runtime_id=" << last_runtime_id
                                  << ") was not delivered; ensure that the finish-time RT-profiler flush is emitted";

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, TraceReplayResolvesKernelSources) {
    constexpr int kDeviceId = 0;
    constexpr uint32_t kWarmupRuntimeId = 0x6001;
    constexpr uint32_t kTraceRuntimeId = 0x6002;
    constexpr size_t kTraceRegionSize = 8 * 1024 * 1024;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, kTraceRegionSize, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&records](const ProgramRealtimeRecordBatch& batch) {
            records.insert(records.end(), batch.records.begin(), batch.records.end());
        });

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
    for (auto& [_, prog] : workload.get_programs()) {
        prog.set_runtime_id(static_cast<uint64_t>(kTraceRuntimeId));
    }

    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), mesh_cq.id());
    distributed::EnqueueMeshWorkload(mesh_cq, workload, false);
    mesh_device->end_mesh_trace(mesh_cq.id(), trace_id);
    mesh_device->replay_mesh_trace(mesh_cq.id(), trace_id, true);

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(trace_id);

    const std::string expected_marker = kSourceMarkerPrefix + std::to_string(kTraceRuntimeId);
    uint32_t trace_records = 0;
    for (const auto& rec : records) {
        if (rec.runtime_id != kTraceRuntimeId) {
            continue;
        }
        ++trace_records;
        ASSERT_FALSE(rec.kernel_sources.empty())
            << "Trace-replayed record (runtime_id=" << kTraceRuntimeId
            << ") carried no kernel sources; its runtime_id was not tied during trace capture";
        for (const auto& src : rec.kernel_sources) {
            EXPECT_NE(src.find(expected_marker), std::string_view::npos)
                << "Trace-replayed record resolved to the wrong program's source: " << src;
        }
    }
    EXPECT_GT(trace_records, 0u) << "No records observed for the trace-replayed program (runtime_id=" << kTraceRuntimeId
                                 << ")";

    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
