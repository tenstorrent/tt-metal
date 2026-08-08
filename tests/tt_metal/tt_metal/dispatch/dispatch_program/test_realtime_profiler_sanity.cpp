// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merge-gate sanity check for the real-time (RT) profiler on Wormhole and
// Blackhole single-chip configurations. Enqueues a handful of compute
// programs back-to-back on all tensix cores, attaches an RT profiler
// callback, and asserts that each program produces a record with a
// plausible start/end timestamp. The goal is to catch coarse regressions
// in the RT profiler pipeline (mailbox layout, D2H socket init, clock
// sync, kernel source propagation, timestamp extraction) before they
// reach CI's longer-running profiler test suite.
//
// Lives in the dispatch "basic" test library so it runs as part of
// `tt-metalium-validation-basic`, which the merge-gate `metalium-basic-tests`
// job executes on both N150 (WH) and P150b (BH). On configs where RT
// profiler cannot be enabled (ETH dispatch, non-MMIO chip, kernels
// nullified, IOMMU-off on BH, etc.) the test skips gracefully via
// IsProgramRealtimeProfilerActive().

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <tt-logger/tt-logger.hpp>

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

namespace tt::tt_metal {
namespace {

using namespace std::chrono_literals;
using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
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

Program make_sanity_program(const std::string& kernel_src, const CoreRange& cores, uint32_t runtime_id) {
    Program program = CreateProgram();
    CreateKernelFromString(
        program,
        kernel_src,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernelFromString(
        program,
        kernel_src,
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernelFromString(program, kernel_src, cores, ComputeConfig{});
    program.set_runtime_id(static_cast<uint64_t>(runtime_id));
    return program;
}

void enqueue_rt_program(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Program program, bool blocking) {
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, blocking);
}

void enqueue_sanity_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t runtime_id, const CoreRange& cores) {
    enqueue_rt_program(
        mesh_device, make_sanity_program(make_sanity_kernel_source(runtime_id), cores, runtime_id), /*blocking=*/false);
}

std::shared_ptr<distributed::MeshDevice> open_unit_mesh(size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        /*device_id=*/0,
        DEFAULT_L1_SMALL_SIZE,
        trace_region_size,
        /*num_command_queues=*/1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    if (mesh_device == nullptr) {
        return nullptr;
    }
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        return nullptr;
    }
    return mesh_device;
}

CoreRange all_cores(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    const CoreCoord grid = mesh_device->compute_with_storage_grid_size();
    return CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1});
}

void enqueue_programs(const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t count) {
    // Runtime IDs start at 1 so every program emits a record (runtime_id == 0
    // is reserved for infrastructure traffic and filtered host-side).
    for (uint32_t i = 1; i <= count; ++i) {
        enqueue_sanity_program(mesh_device, i, all_cores(mesh_device));
    }
}

template <typename Predicate>
void quiesce_and_wait_for(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Predicate delivered) {
    mesh_device->quiesce_devices();
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!delivered() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
}

TEST(RealtimeProfilerSanity, RecordsAreWellFormedAndCarryTheirProgramsSources) {
    auto mesh_device = open_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    uint64_t dropped = 0;
    std::atomic<size_t> delivered = 0;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        dropped += batch.dropped;
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        delivered.fetch_add(batch.records.size());
    });

    enqueue_programs(mesh_device, kNumPrograms);
    quiesce_and_wait_for(mesh_device, [&] { return delivered.load() >= kNumPrograms; });
    UnregisterProgramRealtimeProfilerCallback(handle);

    ASSERT_GE(records.size(), kNumPrograms)
        << "Expected at least " << kNumPrograms << " RT profiler records (one per program), got " << records.size();
    EXPECT_EQ(dropped, 0u);

    for (const auto& rec : records) {
        EXPECT_GT(rec.end_timestamp, rec.start_timestamp)
            << "RT record end_timestamp must be strictly greater than start_timestamp (runtime_id=" << rec.runtime_id
            << ", chip=" << rec.chip_id << ")";
        EXPECT_GT(rec.frequency, 0.0) << "RT record frequency must be positive (runtime_id=" << rec.runtime_id
                                      << ", chip=" << rec.chip_id << ")";

        EXPECT_GT(rec.clock_sync.error, std::chrono::nanoseconds::zero())
            << "RT record clock-sync error should be populated (runtime_id=" << rec.runtime_id << ")";

        EXPECT_LT(rec.duration().count(), kMaxDurationNs)
            << "RT record duration is implausibly large (runtime_id=" << rec.runtime_id << ", chip=" << rec.chip_id
            << ", duration_ns=" << rec.duration().count() << ")";
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

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, CloseDrainsRegisteredCallback) {
    auto mesh_device = open_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
    });

    enqueue_programs(mesh_device, kNumPrograms);
    mesh_device->quiesce_devices();
    EXPECT_TRUE(mesh_device->close());
    UnregisterProgramRealtimeProfilerCallback(handle);

    std::set<uint32_t> observed;
    for (const auto& rec : records) {
        if (rec.runtime_id >= 1 && rec.runtime_id <= kNumPrograms) {
            observed.insert(rec.runtime_id);
        }
    }
    EXPECT_EQ(observed.size(), kNumPrograms)
        << "Mesh close should drain records for callbacks still registered at shutdown";
}

TEST(RealtimeProfilerSanity, ThrowingCallbackIsIsolated) {
    auto mesh_device = open_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    uint64_t throwing_invocations = 0;
    std::vector<ProgramRealtimeRecord> records;
    const auto throwing_handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch&) {
        ++throwing_invocations;
        throw std::runtime_error("intentional callback failure");
    });
    const auto good_handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
    });

    enqueue_programs(mesh_device, kNumPrograms);
    mesh_device->quiesce_devices();
    // Give the RT profiler receiver thread a moment to drain the last
    // socket pages before we unregister. 500ms mirrors the programming
    // example at test_realtime_profiler_csv.cpp and has proven sufficient
    // for small workloads on WH/BH single-chip.
    std::this_thread::sleep_for(500ms);

    UnregisterProgramRealtimeProfilerCallback(throwing_handle);
    UnregisterProgramRealtimeProfilerCallback(good_handle);

    std::set<uint32_t> observed;
    for (const auto& rec : records) {
        if (rec.runtime_id >= 1 && rec.runtime_id <= kNumPrograms) {
            observed.insert(rec.runtime_id);
        }
    }
    EXPECT_GT(throwing_invocations, 0u) << "throwing callback should have been invoked";
    EXPECT_EQ(observed.size(), kNumPrograms)
        << "sibling callback must receive every record despite the other callback throwing";
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, LastProgramRecordDeliveredOnFinish) {
    auto mesh_device = open_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    std::atomic<bool> saw_last = false;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        for (const auto& rec : batch.records) {
            if (rec.runtime_id == kNumPrograms) {
                saw_last.store(true);
            }
        }
    });

    enqueue_programs(mesh_device, kNumPrograms);
    distributed::Finish(mesh_device->mesh_command_queue());
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!saw_last.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
    UnregisterProgramRealtimeProfilerCallback(handle);

    EXPECT_TRUE(saw_last.load()) << "the final program's record (runtime_id=" << kNumPrograms << ") was not delivered";
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSanity, TraceReplayResolvesKernelSources) {
    constexpr uint32_t kWarmupRuntimeId = 0x6001;
    constexpr uint32_t kTraceRuntimeId = 0x6002;
    constexpr size_t kTraceRegionSize = 8 * 1024 * 1024;

    auto mesh_device = open_unit_mesh(kTraceRegionSize);
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    std::atomic<bool> saw_trace = false;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        for (const auto& rec : batch.records) {
            if (rec.runtime_id == kTraceRuntimeId) {
                saw_trace.store(true);
            }
        }
    });

    const CoreRange cores = all_cores(mesh_device);
    Program program =
        make_sanity_program(make_sanity_kernel_source(kTraceRuntimeId), cores, /*runtime_id=*/kWarmupRuntimeId);

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

    quiesce_and_wait_for(mesh_device, [&] { return saw_trace.load(); });
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

// Mapped host_start/host_end must land inside host brackets around each enqueue.
TEST(RealtimeProfilerSanity, RecordHostTimeFallsInDispatchWindow) {
    auto mesh_device = open_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    std::atomic<size_t> delivered = 0;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        delivered.fetch_add(batch.records.size());
    });

    const CoreRange cores = all_cores(mesh_device);

    const std::string fixed_src = make_sanity_kernel_source(/*runtime_id=*/0);

    auto enqueue_blocking = [&](uint32_t runtime_id) {
        enqueue_rt_program(mesh_device, make_sanity_program(fixed_src, cores, runtime_id), /*blocking=*/true);
    };

    enqueue_blocking(/*runtime_id=*/1);

    struct HostWindow {
        std::chrono::steady_clock::time_point before;
        std::chrono::steady_clock::time_point after;
    };
    std::map<uint32_t, HostWindow> windows;
    constexpr uint32_t kFirstMeasured = 2;
    constexpr uint32_t kNumMeasured = 8;
    for (uint32_t runtime_id = kFirstMeasured; runtime_id < kFirstMeasured + kNumMeasured; ++runtime_id) {
        const auto before = std::chrono::steady_clock::now();
        enqueue_blocking(runtime_id);
        const auto after = std::chrono::steady_clock::now();
        windows[runtime_id] = {before, after};
    }

    quiesce_and_wait_for(mesh_device, [&] { return delivered.load() >= kNumMeasured; });
    UnregisterProgramRealtimeProfilerCallback(handle);

    constexpr auto kSlack = std::chrono::milliseconds(2);

    int checked = 0;
    std::chrono::nanoseconds worst_outside{};
    double min_freq = 0.0;
    double max_freq = 0.0;
    for (const auto& rec : records) {
        auto it = windows.find(rec.runtime_id);
        if (it == windows.end() || rec.frequency <= 0.0) {
            continue;
        }
        const double frequency = rec.frequency;
        min_freq = (checked == 0) ? frequency : std::min(min_freq, frequency);
        max_freq = (checked == 0) ? frequency : std::max(max_freq, frequency);
        const auto host_start = rec.host_start();
        const auto host_end = rec.host_end();
        const HostWindow& window = it->second;

        EXPECT_GE(host_start, window.before - kSlack)
            << "runtime_id=" << rec.runtime_id << ": record host start precedes the dispatch window start by "
            << (window.before - host_start).count() << "ns";
        EXPECT_LE(host_end, window.after + kSlack)
            << "runtime_id=" << rec.runtime_id << ": record host end follows the dispatch window end by "
            << (host_end - window.after).count() << "ns";

        worst_outside = std::max({worst_outside, window.before - host_start, host_end - window.after});
        ++checked;
    }

    EXPECT_GT(checked, 0) << "No RT records matched a measured dispatch window";
    log_info(
        tt::LogTest,
        "[RT profiler sanity] checked {} record(s); worst excursion outside dispatch window = {}ns (<= 0 means fully "
        "inside); record frequency = [{}, {}] GHz",
        checked,
        worst_outside.count(),
        min_freq,
        max_freq);
    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
