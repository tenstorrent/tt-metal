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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unistd.h>
#include <utility>
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
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/experimental/realtime_profiler.hpp>
#include <tt-metalium/host_buffer.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/realtime_profiler/realtime_profiler_service.hpp"
#include "distributed/mesh_device_impl.hpp"

namespace tt::tt_metal {
namespace {

using namespace std::chrono_literals;
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

// clock_sync.sync_error_ns is half the sync-handshake RTT — the minimax bound on the offset anchor, and what each
// record self-reports as its sync accuracy. Healthy is ~1-2µs; the servo rejects re-anchors whose half-RTT exceeds
// ~25µs, so the accepted distribution stays well under that. SyncAccuracy asserts the p50/p90/p99 across a session
// rather than a single loose per-record ceiling (which said nothing about the actual distribution).
constexpr uint64_t kSyncErrorP50Ns = 6'000;
constexpr uint64_t kSyncErrorP90Ns = 10'000;
constexpr uint64_t kSyncErrorP99Ns = 15'000;

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

// The two tests below drive RealtimeProfilerService directly instead of opening a mesh, because neither property can be
// observed through a device. A MeshDevice contributes exactly one record ring, so no device test can show a consumer
// servicing several of them; and blocking inside a callback to catch unregister mid-flight needs a ring the test drives
// by hand. Everything else about the service and its ring is covered by the device tests below and by
// tests/tt_metal/tt_metal/misc/test_broadcast_ring.cpp.
experimental::ProgramRealtimeRecord make_service_record(uint32_t runtime_id, uint32_t chip_id) {
    return experimental::ProgramRealtimeRecord{
        .runtime_id = runtime_id,
        .chip_id = chip_id,
        .start_timestamp = runtime_id * 10,
        .end_timestamp = runtime_id * 10 + 5,
        .frequency = 1.0,
        .clock_sync = {.device_cycle_offset = 0, .sync_error_ns = 0},
        .kernel_sources = {},
    };
}

// The central claim of the producer/consumer split: each consumer gets one delivery thread, and that single thread
// services every attached ring. A multi-MeshDevice run depends on this to fan records from all meshes into one
// callback.
TEST(RealtimeProfilerSanity, OneConsumerThreadReadsEveryAttachedRing) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring_a(16);
    RealtimeProfilerRecordRing ring_b(16);

    std::mutex mutex;
    std::condition_variable cv;
    std::set<uint32_t> received_runtime_ids;
    std::set<std::thread::id> callback_threads;
    const auto handle = service.register_consumer([&](const ProgramRealtimeRecordBatch& batch) {
        {
            std::lock_guard lock(mutex);
            callback_threads.insert(std::this_thread::get_id());
            for (const auto& record : batch.records) {
                received_runtime_ids.insert(record.runtime_id);
            }
        }
        cv.notify_all();
    });

    service.attach_ring(ring_a, 8);
    service.attach_ring(ring_b, 8);
    const experimental::ProgramRealtimeRecord records_a[] = {make_service_record(1, 1), make_service_record(2, 1)};
    const experimental::ProgramRealtimeRecord records_b[] = {make_service_record(3, 2), make_service_record(4, 2)};
    ring_a.writer().publish_batch(records_a);
    ring_b.writer().publish_batch(records_b);
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return received_runtime_ids.size() == 4; }))
            << "records from both rings should reach the consumer";
        EXPECT_EQ(received_runtime_ids, (std::set<uint32_t>{1, 2, 3, 4}));
        EXPECT_EQ(callback_threads.size(), 1u) << "a consumer must be served by exactly one delivery thread";
    }

    service.detach_ring(ring_a);
    service.detach_ring(ring_b);
    service.unregister_consumer(handle);
}

// UnregisterProgramRealtimeProfilerCallback documents that it blocks until any in-flight invocation of that callback
// has returned, so a caller can free state the callback captured as soon as it returns.
TEST(RealtimeProfilerSanity, UnregisterWaitsForInFlightCallback) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(8);
    service.attach_ring(ring, 4);

    std::mutex mutex;
    std::condition_variable cv;
    bool callback_started = false;
    bool release_callback = false;
    const auto handle = service.register_consumer([&](const ProgramRealtimeRecordBatch&) {
        std::unique_lock lock(mutex);
        callback_started = true;
        cv.notify_all();
        cv.wait(lock, [&] { return release_callback; });
    });

    ring.writer().publish(make_service_record(1, 1));
    service.wake_consumers();
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return callback_started; }));
    }

    std::atomic<bool> unregister_returned = false;
    std::thread unregister_thread([&] {
        service.unregister_consumer(handle);
        unregister_returned.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(unregister_returned.load(std::memory_order_acquire))
        << "unregister must not return while the callback is still running";

    {
        std::lock_guard lock(mutex);
        release_callback = true;
    }
    cv.notify_all();
    unregister_thread.join();
    EXPECT_TRUE(unregister_returned.load(std::memory_order_acquire));
    service.detach_ring(ring);
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

        EXPECT_GT(rec.clock_sync.sync_error_ns, 0u)
            << "RT record sync_error_ns should be set by the init sync handshake (runtime_id=" << rec.runtime_id << ")";

        if (rec.frequency > 0.0 && rec.end_timestamp > rec.start_timestamp) {
            const double duration_ns = rec.duration_ns();
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

// Sync accuracy is what each record self-reports as clock_sync.sync_error_ns. Spread programs across many servo
// re-anchor intervals so the records sample that value over the session, then assert its distribution stays tight.
// A degraded handshake (slow/contended PCIe) lifts the median; a servo that stops rejecting bad re-anchors fattens the
// tail — p50/p90/p99 catch each, unlike a single per-record ceiling.
TEST(RealtimeProfilerSanity, SyncAccuracy) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::mutex records_mu;
    std::vector<uint64_t> sync_errors_ns;
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
            std::lock_guard<std::mutex> lk(records_mu);
            for (const auto& rec : batch.records) {
                sync_errors_ns.push_back(rec.clock_sync.sync_error_ns);
            }
        });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});
    // A fixed runtime_id keeps the kernel source (and its JIT compile) shared across iterations; the 50ms spacing
    // straddles kServoInterval so successive records fall in different re-anchor epochs.
    constexpr uint32_t kIterations = 40;
    for (uint32_t i = 0; i < kIterations; ++i) {
        enqueue_sanity_program(mesh_device, /*runtime_id=*/1, all_cores);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    UnregisterProgramRealtimeProfilerCallback(handle);

    std::lock_guard<std::mutex> lk(records_mu);
    ASSERT_GE(sync_errors_ns.size(), kIterations / 2)
        << "too few records (" << sync_errors_ns.size() << ") to characterize the sync-error distribution";
    std::sort(sync_errors_ns.begin(), sync_errors_ns.end());
    const auto pct = [&](double p) {
        const size_t idx = static_cast<size_t>(std::lround(p * static_cast<double>(sync_errors_ns.size() - 1)));
        return sync_errors_ns[std::min(sync_errors_ns.size() - 1, idx)];
    };
    const uint64_t p50 = pct(0.50);
    const uint64_t p90 = pct(0.90);
    const uint64_t p99 = pct(0.99);
    std::cout << "[ SYNC ] sync_error_ns p50=" << p50 << " p90=" << p90 << " p99=" << p99
              << " ns (n=" << sync_errors_ns.size() << ")" << std::endl;

    EXPECT_GT(p50, 0u) << "sync_error_ns should be populated by the sync handshake";
    EXPECT_LT(p50, kSyncErrorP50Ns) << "median sync error too high; the handshake is systematically degraded";
    EXPECT_LT(p90, kSyncErrorP90Ns) << "p90 sync error too high";
    EXPECT_LT(p99, kSyncErrorP99Ns) << "tail sync error too high; a bad re-anchor is not being rejected";

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

// Independent host/device sync-accuracy check. The Tracy SYNC_CHECK markers are a self-consistency check — their host
// and device endpoints are both derived from the same calibration, so they coincide by construction and cannot reveal
// a wrong mapping. This test instead brackets each program's on-device execution between two host-clock reads and
// asserts the record's device->host mapping — reconstructed solely from the record's own device_cycle_offset and
// frequency — lands inside that independently-measured host window. A mis-signed/mis-scaled offset, a wrong clock
// domain, or a stale anchor would push the record outside the window. This is only possible because the affine mapping
// now rides on the record itself.
TEST(RealtimeProfilerSanity, RecordHostTimeFallsInDispatchWindow) {
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

    // One fixed kernel source for every program so the JIT ELF is compiled once (during warm-up) and reused; this
    // keeps each measured dispatch window free of multi-second kernel-compilation time.
    const std::string fixed_src = make_sanity_kernel_source(/*runtime_id=*/0);

    auto enqueue_blocking = [&](uint32_t runtime_id) {
        Program program = CreateProgram();
        CreateKernelFromString(
            program,
            fixed_src,
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        CreateKernelFromString(
            program,
            fixed_src,
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        CreateKernelFromString(program, fixed_src, all_cores, ComputeConfig{});
        program.set_runtime_id(static_cast<uint64_t>(runtime_id));
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
    };

    // Warm up the JIT cache; runtime_id 1 is not placed in the window map.
    enqueue_blocking(/*runtime_id=*/1);

    struct HostWindow {
        int64_t before_ticks;
        int64_t after_ticks;
    };
    std::map<uint32_t, HostWindow> windows;
    const auto host_ns = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    constexpr uint32_t kFirstMeasured = 2;
    constexpr uint32_t kNumMeasured = 8;
    for (uint32_t runtime_id = kFirstMeasured; runtime_id < kFirstMeasured + kNumMeasured; ++runtime_id) {
        const int64_t before = host_ns();
        enqueue_blocking(runtime_id);  // blocks until the program completes on device
        const int64_t after = host_ns();
        windows[runtime_id] = {before, after};
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    UnregisterProgramRealtimeProfilerCallback(handle);

    // Generous bound: it covers host-clock read jitter and the gap between the bracketing reads and the actual
    // dispatch/completion. The check catches a fundamentally wrong mapping (off by ms and up), not µs-level skew.
    constexpr double kSlackNs = 2'000'000.0;

    int checked = 0;
    double worst_outside_ns = 0.0;
    double min_freq = 0.0;
    double max_freq = 0.0;
    std::lock_guard<std::mutex> lk(records_mu);
    for (const auto& rec : records) {
        auto it = windows.find(rec.runtime_id);
        if (it == windows.end() || rec.frequency <= 0.0) {
            continue;
        }
        const double frequency = rec.frequency;
        min_freq = (checked == 0) ? frequency : std::min(min_freq, frequency);
        max_freq = (checked == 0) ? frequency : std::max(max_freq, frequency);
        const double host_start_ns = static_cast<double>(rec.host_start_ns());
        const double host_end_ns = static_cast<double>(rec.host_end_ns());
        const double before_ns = static_cast<double>(it->second.before_ticks);
        const double after_ns = static_cast<double>(it->second.after_ticks);

        EXPECT_GE(host_start_ns, before_ns - kSlackNs)
            << "runtime_id=" << rec.runtime_id << ": record host start " << host_start_ns
            << "ns precedes the dispatch window start " << before_ns << "ns";
        EXPECT_LE(host_end_ns, after_ns + kSlackNs)
            << "runtime_id=" << rec.runtime_id << ": record host end " << host_end_ns
            << "ns follows the dispatch window end " << after_ns << "ns";

        worst_outside_ns = std::max({worst_outside_ns, before_ns - host_start_ns, host_end_ns - after_ns});
        ++checked;
    }

    EXPECT_GT(checked, 0) << "No RT records matched a measured dispatch window";
    // <= 0 means every record mapped strictly inside its window; a small positive value is measurement jitter.
    std::cout << "[ SYNC ] checked " << checked
              << " record(s); worst excursion outside dispatch window = " << worst_outside_ns
              << "ns (<= 0 means fully inside); record frequency = [" << min_freq << ", " << max_freq << "] GHz"
              << std::endl;

    EXPECT_TRUE(mesh_device->close());
}

// Independent measurement of the RT profiler's clock-sync accuracy, sharing no code with the production sync. The
// roles are deliberately inverted: here the *device* times a round trip and brackets a host clock read inside it,
// where RealtimeProfilerClockSync has the host time a round trip and bracket a device timestamp. A wrong anchor
// placement, a wrong clock domain, or a mis-signed offset in production therefore cannot reproduce itself here.
//
// Per iteration the kernel stamps WALL_CLOCK, asks the host for a reply, and stamps again once the reply lands. The
// host's clock read happened somewhere inside that device-measured interval, so (host read, interval midpoint) samples
// the same mapping. Bracketing on the device keeps the host's polling cost out of the interval; what is left is the
// asymmetry between the two legs, which the midpoint splits and the reported bracket bounds. Only the tightest
// brackets are kept, those being the round trips that queued least on either leg.
constexpr uint32_t kSyncProbeIterations = 2000;
constexpr double kSyncProbeKeepFraction = 0.05;

std::string make_sync_kernel_source() {
    return R"(
#include <cstdint>
#include "risc_common.h"

void kernel_main() {
    const uint32_t ack_addr = get_arg_val<uint32_t>(0);
    const uint32_t request_addr = get_arg_val<uint32_t>(1);
    const uint32_t log_addr = get_arg_val<uint32_t>(2);
    const uint32_t iterations = get_arg_val<uint32_t>(3);
    // PinnedMemory hands the host the address only; a NOC write also needs the PCIe endpoint's coordinates, which the
    // JIT build supplies as PCIE_NOC_X/Y. Same encoding cq_realtime_profiler_push.cpp builds.
    const uint64_t host_addr =
        uint64_t(NOC_XY_PCIE_ENCODING(NOC_X_PHYS_COORD(PCIE_NOC_X), NOC_Y_PHYS_COORD(PCIE_NOC_Y))) |
        ((static_cast<uint64_t>(get_arg_val<uint32_t>(5)) << 32) | static_cast<uint64_t>(get_arg_val<uint32_t>(4)));

    volatile tt_l1_ptr uint32_t* ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ack_addr);
    (void)ack;
    volatile tt_l1_ptr uint32_t* request = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(request_addr);
    volatile tt_l1_ptr uint32_t* log = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(log_addr);

    for (uint32_t i = 1; i <= iterations; i++) {
        request[0] = i;
        const uint64_t sent = get_timestamp();
        noc_async_write(request_addr, host_addr, sizeof(uint32_t));
        noc_async_write_barrier();
        const uint64_t written = get_timestamp();
        while (NOC_STREAM_READ_REG(0, STREAM_SCRATCH_REG_INDEX) != i) {
        }
        const uint64_t answered = get_timestamp();
        log[8 * (i - 1) + 4] = static_cast<uint32_t>(written & 0xFFFFFFFF);
        log[8 * (i - 1) + 5] = static_cast<uint32_t>(written >> 32);
        log[8 * (i - 1) + 0] = static_cast<uint32_t>(sent & 0xFFFFFFFF);
        log[8 * (i - 1) + 1] = static_cast<uint32_t>(sent >> 32);
        log[8 * (i - 1) + 2] = static_cast<uint32_t>(answered & 0xFFFFFFFF);
        log[8 * (i - 1) + 3] = static_cast<uint32_t>(answered >> 32);
    }
}
)";
}

struct SyncProbe {
    int64_t host_ns = 0;
    double device_mid_ticks = 0.0;
    uint64_t bracket_ticks = 0;
};

TEST(RealtimeProfilerSanity, SyncAccuracyAgainstIndependentHandshake) {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    if (!IsProgramRealtimeProfilerActive()) {
        GTEST_SKIP() << "Real-time profiler is not active on this configuration";
    }

    std::mutex records_mu;
    std::vector<ProgramRealtimeRecord> records;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        std::lock_guard lk(records_mu);
        records.insert(records.end(), batch.records.begin(), batch.records.end());
    });

    IDevice* device = mesh_device->get_devices().front();
    const uint32_t l1_base = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t ack_addr = l1_base;
    const uint32_t request_addr = l1_base + 16;
    const uint32_t log_addr = l1_base + 32;
    const CoreCoord core{0, 0};
    const CoreCoord vcore = device->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto& cluster = MetalContext::instance().get_cluster();
    // Stream 0 scratch register: plain read/write storage (scratch exists only in streams 0-3 and 8-11, and 8-39 are
    // the CB range), unused on this core. Polling it costs the kernel a register load with no cache in the way.
    constexpr uint32_t kStreamScratchRegIndex = 36;
    const uint32_t stream_scratch_addr =
        MetalContext::instance().hal().get_noc_overlay_start_addr() + kStreamScratchRegIndex * sizeof(uint32_t);
    {
        const std::vector<uint32_t> zeros(8, 0);
        cluster.write_core(zeros.data(), zeros.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), vcore), ack_addr);
    }

    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    std::shared_ptr<uint32_t[]> request_backing(
        static_cast<uint32_t*>(std::aligned_alloc(page_size, page_size)), [](uint32_t* p) { std::free(p); });
    ASSERT_NE(request_backing, nullptr);
    request_backing[0] = 0;
    HostBuffer request_view(ttsl::Span<uint32_t>(request_backing.get(), 4), MemoryPin(request_backing));
    distributed::MeshCoordinateRangeSet request_range;
    request_range.merge(distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0)));
    auto request_pinned = experimental::PinnedMemory::Create(*mesh_device, request_range, request_view, true);
    if (!request_pinned || !request_pinned->get_noc_addr(device->id()).has_value()) {
        UnregisterProgramRealtimeProfilerCallback(handle);
        GTEST_SKIP() << "Host memory cannot be pinned for device writes on this configuration";
    }
    const uint64_t request_noc_addr = request_pinned->get_noc_addr(device->id())->addr;
    volatile uint32_t* request_word = request_backing.get();
    volatile uint32_t* ready_word_probe = request_backing.get() + 3;

    Program program = CreateProgram();
    auto kernel = CreateKernelFromString(
        program,
        make_sync_kernel_source(),
        CoreRangeSet(CoreRange(core, core)),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernelFromString(
        program, "#include <cstdint>\nvoid kernel_main() {}\n", CoreRangeSet(CoreRange(core, core)), ComputeConfig{});
    SetRuntimeArgs(
        program,
        kernel,
        core,
        {ack_addr,
         request_addr,
         log_addr,
         kSyncProbeIterations,
         static_cast<uint32_t>(request_noc_addr & 0xFFFFFFFFull),
         static_cast<uint32_t>(request_noc_addr >> 32)});
    program.set_runtime_id(9001);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);

    constexpr uint32_t kCalibrationReps = 2000;
    auto min_write_ns = std::chrono::nanoseconds::max();
    for (uint32_t r = 0; r < kCalibrationReps; r++) {
        const uint32_t probe = 0;
        const auto before = std::chrono::steady_clock::now();
        cluster.write_core_immediate(&probe, sizeof(probe), tt_cxy_pair(device->id(), vcore), log_addr + 4096);
        min_write_ns = std::min(min_write_ns, std::chrono::steady_clock::now() - before);
    }
    auto min_detect_ns = std::chrono::nanoseconds::max();
    for (uint32_t r = 0; r < kCalibrationReps; r++) {
        const auto before = std::chrono::steady_clock::now();
        const uint32_t observed = ready_word_probe[0];
        const auto delta = std::chrono::steady_clock::now() - before;
        if (observed == 0xFFFFFFFF) {
            continue;
        }
        min_detect_ns = std::min(min_detect_ns, delta);
    }

    std::vector<int64_t> host_ns_by_iteration;
    host_ns_by_iteration.reserve(kSyncProbeIterations);
    for (uint32_t i = 1; i <= kSyncProbeIterations; i++) {
        bool asked = false;
        for (uint32_t poll = 0; poll < 2000000; poll++) {
            if (request_word[0] == i) {
                asked = true;
                break;
            }
        }
        if (!asked) {
            break;
        }
        host_ns_by_iteration.push_back(std::chrono::steady_clock::now().time_since_epoch().count());
        cluster.write_core_immediate(&i, sizeof(i), tt_cxy_pair(device->id(), vcore), stream_scratch_addr);
    }

    distributed::Finish(mesh_device->mesh_command_queue());
    std::this_thread::sleep_for(500ms);
    UnregisterProgramRealtimeProfilerCallback(handle);

    ASSERT_GE(host_ns_by_iteration.size(), 100u)
        << "kernel stopped asking after " << host_ns_by_iteration.size() << " iterations";

    const std::vector<uint32_t> log = cluster.read_core(
        device->id(), vcore, log_addr, static_cast<uint32_t>(host_ns_by_iteration.size() * 8 * sizeof(uint32_t)));
    std::vector<double> device_work_ns;
    std::vector<double> remainder_ns;
    std::vector<SyncProbe> probes;
    probes.reserve(host_ns_by_iteration.size());
    for (size_t i = 0; i < host_ns_by_iteration.size(); i++) {
        const uint64_t sent = (static_cast<uint64_t>(log[8 * i + 1]) << 32) | log[8 * i + 0];
        const uint64_t answered = (static_cast<uint64_t>(log[8 * i + 3]) << 32) | log[8 * i + 2];
        const uint64_t written = (static_cast<uint64_t>(log[8 * i + 5]) << 32) | log[8 * i + 4];
        if (written > sent && answered > written) {
            device_work_ns.push_back(static_cast<double>(written - sent));
            remainder_ns.push_back(static_cast<double>(answered - written));
        }
        if (answered <= sent) {
            continue;
        }
        probes.push_back(SyncProbe{
            .host_ns = host_ns_by_iteration[i],
            .device_mid_ticks = static_cast<double>(sent) + static_cast<double>(answered - sent) / 2.0,
            .bracket_ticks = (answered - sent) / 2});
    }
    ASSERT_GE(probes.size(), 100u) << "too few usable probes: " << probes.size();

    std::sort(probes.begin(), probes.end(), [](const SyncProbe& a, const SyncProbe& b) {
        return a.bracket_ticks < b.bracket_ticks;
    });
    probes.resize(std::max<size_t>(16, static_cast<size_t>(probes.size() * kSyncProbeKeepFraction)));

    double mean_host = 0.0;
    double mean_device = 0.0;
    for (const auto& p : probes) {
        mean_host += static_cast<double>(p.host_ns);
        mean_device += p.device_mid_ticks;
    }
    mean_host /= static_cast<double>(probes.size());
    mean_device /= static_cast<double>(probes.size());
    double num = 0.0;
    double den = 0.0;
    for (const auto& p : probes) {
        const double dx = static_cast<double>(p.host_ns) - mean_host;
        num += dx * (p.device_mid_ticks - mean_device);
        den += dx * dx;
    }
    ASSERT_GT(den, 0.0) << "probes span no host time";
    const double frequency = num / den;
    const double offset = mean_device - frequency * mean_host;

    std::lock_guard lk(records_mu);
    ASSERT_FALSE(records.empty()) << "no RT records collected";
    const double worst_bracket_ns = static_cast<double>(probes.back().bracket_ticks) / frequency;
    // The two mappings sit a fixed distance apart, since this test splits its round trip at the midpoint while its two
    // legs are not equal. That constant is bounded by its own bracket; what has to hold is that it does not move,
    // because both mappings track the same device clock.
    std::vector<double> disagreements;
    uint64_t reported_error_ns = 0;
    for (const auto& rec : records) {
        if (rec.frequency <= 0.0) {
            continue;
        }
        const double independent_host_ns = (static_cast<double>(rec.start_timestamp) - offset) / frequency;
        disagreements.push_back(static_cast<double>(rec.host_start_ns()) - independent_host_ns);
        reported_error_ns = std::max(reported_error_ns, rec.clock_sync.sync_error_ns);
    }
    ASSERT_FALSE(disagreements.empty());
    const auto [min_it, max_it] = std::minmax_element(disagreements.begin(), disagreements.end());
    const double offset_spread_ns = *max_it - *min_it;
    const double mean_offset_ns =
        std::accumulate(disagreements.begin(), disagreements.end(), 0.0) / static_cast<double>(disagreements.size());

    const auto tick_ns = [&](const std::vector<double>& v) {
        auto sorted = v;
        std::sort(sorted.begin(), sorted.end());
        return sorted.empty() ? 0.0 : sorted[sorted.size() / 20] / frequency;
    };
    const double device_work = tick_ns(device_work_ns);
    const double remainder = tick_ns(remainder_ns);
    std::cout << "[ BREAKDOWN ] round trip " << (device_work + remainder) << "ns = device write+barrier " << device_work
              << "ns + remainder " << remainder << "ns; of that remainder, host write issue " << min_write_ns.count()
              << "ns + host spin detect " << min_detect_ns.count() << "ns => both flights "
              << (remainder - static_cast<double>(min_write_ns.count()) - static_cast<double>(min_detect_ns.count()))
              << "ns" << std::endl;
    std::cout << "[ SYNC ] device-bracketed handshake: " << probes.size() << " tightest probes, frequency " << frequency
              << " GHz (record reports " << records.front().frequency << "), own bracket +/-" << worst_bracket_ns
              << "ns; RT mapping offset by " << mean_offset_ns << "ns, moving " << offset_spread_ns
              << "ns across the session; RT profiler reports " << reported_error_ns << "ns" << std::endl;

    EXPECT_LE(std::abs(mean_offset_ns), worst_bracket_ns + static_cast<double>(reported_error_ns))
        << "RT mapping sits further from the independent one than both stated uncertainties allow";
    EXPECT_LE(offset_spread_ns, static_cast<double>(reported_error_ns) + worst_bracket_ns / 4.0)
        << "the gap between the two mappings moved during the session; one of them is not tracking the device clock";

    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
