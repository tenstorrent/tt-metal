// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merge-gate sanity check for the real-time (RT) profiler on Wormhole and
// Blackhole single-chip configurations. Enqueues a handful of compute
// programs back-to-back on all tensix cores, attaches an RT profiler
// callback, and asserts that each program produces a record with a
// plausible start/end timestamp. The goal is to catch coarse regressions
// in the RT profiler pipeline (mailbox layout, D2H socket init, sync
// handshake, kernel source propagation, timestamp extraction) before they
// reach CI's longer-running profiler test suite.
//
// Lives in the dispatch "basic" test library so it runs as part of
// `tt-metalium-validation-basic`, which the merge-gate `metalium-basic-tests`
// job executes on both N150 (WH) and P150b (BH). On configs where RT
// profiler cannot be enabled (ETH dispatch, non-MMIO chip, kernels
// nullified, IOMMU-off on BH, etc.) the test skips gracefully via
// IsProgramRealtimeProfilerActive().

#include <atomic>
#include <chrono>
#include <cstdint>
#include <set>
#include <span>
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
#include "tt_metal/impl/dispatch/data_collector.hpp"

namespace tt::tt_metal {
namespace {

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

ProgramRealtimeRecord make_callback_record(uint32_t runtime_id) {
    return ProgramRealtimeRecord{
        .runtime_id = runtime_id,
        .chip_id = 7,
        .start_timestamp = 100 + runtime_id,
        .end_timestamp = 200 + runtime_id,
        .frequency = 1.5,
        .kernel_sources = {},
    };
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

TEST(RealtimeProfilerSanity, DataCollectorNotifiesCallbackListeners) {
    struct Listener : tt::RealtimeProfilerCallbackListener {
        void on_callback_registered(
            ProgramRealtimeProfilerCallbackHandle handle, const ProgramRealtimeProfilerCallback& callback) override {
            added_handle = handle;
            callback(ProgramRealtimeRecordBatch{std::span<const ProgramRealtimeRecord>(&record, 1), 0});
            add_count++;
        }
        void on_callback_unregistered(ProgramRealtimeProfilerCallbackHandle handle) override {
            removed_handle = handle;
            remove_count++;
        }

        ProgramRealtimeRecord record = make_callback_record(7);
        ProgramRealtimeProfilerCallbackHandle added_handle = 0;
        ProgramRealtimeProfilerCallbackHandle removed_handle = 0;
        uint32_t add_count = 0;
        uint32_t remove_count = 0;
    };

    DataCollector collector;
    std::atomic<uint64_t> received{0};
    std::atomic<uint64_t> runtime_sum{0};
    Listener listener;
    collector.AttachRealtimeProfilerCallbackListener(&listener);

    auto handle = collector.RegisterProgramRealtimeProfilerCallback(
        [&received, &runtime_sum](const ProgramRealtimeRecordBatch& batch) {
            uint64_t local_sum = 0;
            for (const auto& record : batch.records) {
                local_sum += record.runtime_id;
            }
            runtime_sum.fetch_add(local_sum, std::memory_order_relaxed);
            received.fetch_add(batch.records.size(), std::memory_order_release);
        });

    EXPECT_EQ(listener.add_count, 1u);
    EXPECT_EQ(listener.added_handle, handle);
    EXPECT_EQ(received.load(std::memory_order_acquire), 1u);
    EXPECT_EQ(runtime_sum.load(std::memory_order_relaxed), 7u);
    collector.UnregisterProgramRealtimeProfilerCallback(handle);
    EXPECT_EQ(listener.remove_count, 1u);
    EXPECT_EQ(listener.removed_handle, handle);
    collector.DetachRealtimeProfilerCallbackListener(&listener);
}

TEST(RealtimeProfilerSanity, DataCollectorReplaysExistingCallbacksWhenListenerAttaches) {
    struct Listener : tt::RealtimeProfilerCallbackListener {
        void on_callback_registered(
            ProgramRealtimeProfilerCallbackHandle handle, const ProgramRealtimeProfilerCallback& callback) override {
            added_handle = handle;
            callback(ProgramRealtimeRecordBatch{std::span<const ProgramRealtimeRecord>(&record, 1), 0});
            add_count++;
        }
        void on_callback_unregistered(ProgramRealtimeProfilerCallbackHandle handle) override {
            removed_handle = handle;
            remove_count++;
        }

        ProgramRealtimeRecord record = make_callback_record(11);
        ProgramRealtimeProfilerCallbackHandle added_handle = 0;
        ProgramRealtimeProfilerCallbackHandle removed_handle = 0;
        uint32_t add_count = 0;
        uint32_t remove_count = 0;
    };

    DataCollector collector;
    std::atomic<uint64_t> received{0};
    std::atomic<uint64_t> runtime_sum{0};

    auto handle = collector.RegisterProgramRealtimeProfilerCallback(
        [&received, &runtime_sum](const ProgramRealtimeRecordBatch& batch) {
            uint64_t local_sum = 0;
            for (const auto& record : batch.records) {
                local_sum += record.runtime_id;
            }
            runtime_sum.fetch_add(local_sum, std::memory_order_relaxed);
            received.fetch_add(batch.records.size(), std::memory_order_release);
        });

    Listener listener;
    collector.AttachRealtimeProfilerCallbackListener(&listener);

    EXPECT_EQ(listener.add_count, 1u);
    EXPECT_EQ(listener.added_handle, handle);
    EXPECT_EQ(received.load(std::memory_order_acquire), 1u);
    EXPECT_EQ(runtime_sum.load(std::memory_order_relaxed), 11u);

    collector.DetachRealtimeProfilerCallbackListener(&listener);
    collector.UnregisterProgramRealtimeProfilerCallback(handle);
    EXPECT_EQ(listener.remove_count, 0u);
}

TEST(RealtimeProfilerSanity, DataCollectorTracksRealtimeProfilerActiveChips) {
    DataCollector collector;

    EXPECT_FALSE(collector.IsRealtimeProfilerActive());
    collector.NotifyRealtimeProfilerActivated(7);
    EXPECT_TRUE(collector.IsRealtimeProfilerActive());
    collector.NotifyRealtimeProfilerActivated(11);
    EXPECT_TRUE(collector.IsRealtimeProfilerActive());
    collector.NotifyRealtimeProfilerDeactivated(7);
    EXPECT_TRUE(collector.IsRealtimeProfilerActive());
    collector.NotifyRealtimeProfilerDeactivated(11);
    EXPECT_FALSE(collector.IsRealtimeProfilerActive());
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

    std::vector<ProgramRealtimeRecord> records_a;
    std::vector<ProgramRealtimeRecord> records_b;
    std::atomic<uint64_t> dropped_a{0};
    std::atomic<uint64_t> dropped_b{0};

    ProgramRealtimeProfilerCallbackHandle handle_a =
        RegisterProgramRealtimeProfilerCallback([&records_a, &dropped_a](const ProgramRealtimeRecordBatch& batch) {
            dropped_a.fetch_add(batch.dropped, std::memory_order_relaxed);
            records_a.insert(records_a.end(), batch.records.begin(), batch.records.end());
        });
    ProgramRealtimeProfilerCallbackHandle handle_b =
        RegisterProgramRealtimeProfilerCallback([&records_b, &dropped_b](const ProgramRealtimeRecordBatch& batch) {
            dropped_b.fetch_add(batch.dropped, std::memory_order_relaxed);
            records_b.insert(records_b.end(), batch.records.begin(), batch.records.end());
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

    UnregisterProgramRealtimeProfilerCallback(handle_a);
    UnregisterProgramRealtimeProfilerCallback(handle_b);

    std::vector<ProgramRealtimeRecord> collected = std::move(records_a);
    std::vector<ProgramRealtimeRecord> collected_b = std::move(records_b);

    ASSERT_GE(collected.size(), kNumPrograms)
        << "Expected at least " << kNumPrograms << " RT profiler records (one per program), got " << collected.size();
    ASSERT_GE(collected_b.size(), kNumPrograms) << "Expected the second callback to receive at least " << kNumPrograms
                                                << " RT profiler records (one per program), got " << collected_b.size();
    EXPECT_EQ(dropped_a.load(std::memory_order_relaxed), 0u);
    EXPECT_EQ(dropped_b.load(std::memory_order_relaxed), 0u);

    for (const auto& rec : collected) {
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
    for (const auto& rec : collected) {
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

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, CloseDrainsRegisteredCallback) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
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
        enqueue_sanity_program(mesh_device, /*runtime_id=*/i + 1, all_cores);
    }

    mesh_device->quiesce_devices();
    EXPECT_TRUE(mesh_device->close());

    std::vector<ProgramRealtimeRecord> collected = std::move(records);
    UnregisterProgramRealtimeProfilerCallback(handle);

    std::set<uint32_t> observed_runtime_ids;
    for (const auto& rec : collected) {
        if (rec.runtime_id >= 1 && rec.runtime_id <= kNumPrograms) {
            observed_runtime_ids.insert(rec.runtime_id);
        }
    }
    EXPECT_EQ(observed_runtime_ids.size(), kNumPrograms)
        << "Mesh close should drain records for callbacks still registered at shutdown";
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

    std::vector<ProgramRealtimeRecord> collected = std::move(records);

    const std::string expected_marker = kSourceMarkerPrefix + std::to_string(kTraceRuntimeId);
    uint32_t trace_records = 0;
    for (const auto& rec : collected) {
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
