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
#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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

#include "impl/realtime_profiler/realtime_profiler_service.hpp"

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

// Runs a single compute program on all tensix cores on `mesh_device`,
// tagged with `runtime_id`, so the RT profiler pipeline emits a record
// carrying that runtime_id (records with runtime_id == 0 are filtered
// out by the host-side receiver).
void enqueue_sanity_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t runtime_id, const CoreRange& cores) {
    enqueue_rt_program(
        mesh_device, make_sanity_program(make_sanity_kernel_source(runtime_id), cores, runtime_id), /*blocking=*/false);
}

// Accumulates everything delivered to one registered callback. Unregistering in the destructor is what makes the
// accumulated records safe to read: the API guarantees no callback is in flight once it returns.
class RecordCollector {
public:
    RecordCollector() {
        handle_ = RegisterProgramRealtimeProfilerCallback([this](const ProgramRealtimeRecordBatch& batch) {
            std::lock_guard lock(mutex_);
            dropped_ += batch.dropped;
            records_.insert(records_.end(), batch.records.begin(), batch.records.end());
        });
        registered_ = true;
    }
    ~RecordCollector() { stop(); }
    RecordCollector(const RecordCollector&) = delete;
    RecordCollector& operator=(const RecordCollector&) = delete;

    void stop() {
        if (std::exchange(registered_, false)) {
            UnregisterProgramRealtimeProfilerCallback(handle_);
        }
    }

    std::vector<ProgramRealtimeRecord> records() const {
        std::lock_guard lock(mutex_);
        return records_;
    }
    uint64_t dropped() const {
        std::lock_guard lock(mutex_);
        return dropped_;
    }
    std::set<uint32_t> runtime_ids_in_range(uint32_t count) const {
        std::set<uint32_t> ids;
        for (const auto& record : records()) {
            if (record.runtime_id >= 1 && record.runtime_id <= count) {
                ids.insert(record.runtime_id);
            }
        }
        return ids;
    }

private:
    mutable std::mutex mutex_;
    std::vector<ProgramRealtimeRecord> records_;
    uint64_t dropped_ = 0;
    ProgramRealtimeProfilerCallbackHandle handle_{};
    bool registered_ = false;
};

class RealtimeProfilerDeviceSanity : public ::testing::Test {
protected:
    void SetUp() override {
        mesh_device_ = distributed::MeshDevice::create_unit_mesh(
            /*device_id=*/0,
            DEFAULT_L1_SMALL_SIZE,
            trace_region_size(),
            /*num_command_queues=*/1,
            DispatchCoreConfig{DispatchCoreType::WORKER});
        ASSERT_NE(mesh_device_, nullptr);
        // Activation flips on during mesh open, so this check is stable by the time
        // create_unit_mesh returns. When it returns false the RT profiler was disabled
        // for this dispatch config (ETH dispatch, non-MMIO chip, kernels nullified, no
        // valid RT core) — treat that as a graceful skip rather than a failure.
        if (!IsProgramRealtimeProfilerActive()) {
            close_mesh();
            GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
        }
    }

    void TearDown() override { close_mesh(); }

    virtual size_t trace_region_size() const { return DEFAULT_TRACE_REGION_SIZE; }

    bool close_mesh() {
        if (mesh_device_ == nullptr) {
            return true;
        }
        const bool closed = mesh_device_->close();
        mesh_device_.reset();
        return closed;
    }

    CoreRange all_cores() const {
        const CoreCoord grid = mesh_device_->compute_with_storage_grid_size();
        return CoreRange(CoreCoord{0, 0}, CoreCoord{grid.x - 1, grid.y - 1});
    }

    void enqueue_programs(uint32_t count) {
        // Runtime IDs start at 1 so every program emits a record (runtime_id == 0
        // is reserved for infrastructure traffic and filtered host-side).
        for (uint32_t i = 1; i <= count; ++i) {
            enqueue_sanity_program(mesh_device_, i, all_cores());
        }
    }

    template <typename Predicate>
    void quiesce_and_wait_for(Predicate delivered) {
        mesh_device_->quiesce_devices();
        const auto deadline = std::chrono::steady_clock::now() + 10s;
        while (!delivered() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(5ms);
        }
    }

    void quiesce_and_wait_for_programs(const RecordCollector& collector, uint32_t count) {
        quiesce_and_wait_for([&] { return collector.runtime_ids_in_range(count).size() >= count; });
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

class RealtimeProfilerDeviceSanityWithTrace : public RealtimeProfilerDeviceSanity {
protected:
    size_t trace_region_size() const override { return 8 * 1024 * 1024; }
};

// Drives RealtimeProfilerService directly rather than opening a mesh: a MeshDevice contributes exactly one record
// ring, so multi-ring delivery, drop accounting and mid-callback control changes are unreachable from a device test.
// The ring itself is covered by tests/tt_metal/tt_metal/misc/test_broadcast_ring.cpp.
experimental::ProgramRealtimeRecord make_service_record(uint32_t runtime_id, uint32_t chip_id) {
    return experimental::ProgramRealtimeRecord{
        .runtime_id = runtime_id,
        .chip_id = chip_id,
        .start_timestamp = runtime_id * 10,
        .end_timestamp = runtime_id * 10 + 5,
        .frequency = 1.0,
        .clock_sync = {.device_cycle_offset = 0, .sync_error = {}},
        .kernel_sources = {},
    };
}

// A producer over a bare ring, so the service can be exercised without standing up a receiver and its devices.
class TestRecordProducer : public ProgramRecordProducer {
public:
    TestRecordProducer(size_t capacity, size_t max_batch) : ring_(capacity), max_batch_records_(max_batch) {}

    size_t max_batch_records() const override { return max_batch_records_; }
    RealtimeProfilerRecordRing::Reader make_reader() override { return ring_.make_reader(); }
    void wait_until_no_readers() override { ring_.wait_until_no_readers(); }
    uint64_t num_published_records() const override { return num_published_records_.load(std::memory_order_relaxed); }

    void publish(std::span<const experimental::ProgramRealtimeRecord> records) {
        ring_.writer().publish_batch(records);
        num_published_records_.fetch_add(records.size(), std::memory_order_relaxed);
    }
    void publish(const experimental::ProgramRealtimeRecord& record) { publish({&record, 1}); }

private:
    RealtimeProfilerRecordRing ring_;
    size_t max_batch_records_;
    std::atomic<uint64_t> num_published_records_{0};
};

// Each consumer gets one delivery thread, and that thread services every attached ring -- what a multi-MeshDevice
// run depends on to fan records from all meshes into one callback.
TEST(RealtimeProfilerSanity, OneConsumerThreadReadsEveryAttachedRing) {
    RealtimeProfilerService service;
    TestRecordProducer ring_a(16, 8);
    TestRecordProducer ring_b(16, 8);

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

    service.attach_producer(ring_a);
    service.attach_producer(ring_b);
    const experimental::ProgramRealtimeRecord records_a[] = {make_service_record(1, 1), make_service_record(2, 1)};
    const experimental::ProgramRealtimeRecord records_b[] = {make_service_record(3, 2), make_service_record(4, 2)};
    ring_a.publish(records_a);
    ring_b.publish(records_b);
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return received_runtime_ids.size() == 4; }))
            << "records from both rings should reach the consumer";
        EXPECT_EQ(received_runtime_ids, (std::set<uint32_t>{1, 2, 3, 4}));
        EXPECT_EQ(callback_threads.size(), 1u) << "a consumer must be served by exactly one delivery thread";
    }

    service.detach_producer(ring_a);
    service.detach_producer(ring_b);
    service.unregister_consumer(handle);
}

// UnregisterProgramRealtimeProfilerCallback blocks until any in-flight invocation of that callback has returned.
TEST(RealtimeProfilerSanity, UnregisterWaitsForInFlightCallback) {
    RealtimeProfilerService service;
    TestRecordProducer ring(8, 4);
    service.attach_producer(ring);

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

    ring.publish(make_service_record(1, 1));
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
    service.detach_producer(ring);
}

// A reader starts at the ring's current head: a late consumer sees the stream from that point on, not the backlog.
TEST(RealtimeProfilerSanity, LateConsumerReceivesOnlyFutureRecords) {
    RealtimeProfilerService service;
    TestRecordProducer ring(8, 4);
    service.attach_producer(ring);

    ring.publish(make_service_record(1, 7));
    service.wake_consumers();

    std::mutex mutex;
    std::condition_variable cv;
    std::vector<uint32_t> received;
    const auto handle = service.register_consumer([&](const ProgramRealtimeRecordBatch& batch) {
        {
            std::lock_guard lock(mutex);
            for (const auto& record : batch.records) {
                received.push_back(record.runtime_id);
            }
        }
        cv.notify_all();
    });

    ring.publish(make_service_record(2, 7));
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return !received.empty(); }));
        EXPECT_EQ(received, (std::vector<uint32_t>{2}));
    }

    service.detach_producer(ring);
    service.unregister_consumer(handle);
}

TEST(RealtimeProfilerSanity, ThrowingConsumerDoesNotAffectSibling) {
    RealtimeProfilerService service;
    TestRecordProducer ring(8, 4);
    service.attach_producer(ring);

    const auto throwing = service.register_consumer(
        [](const ProgramRealtimeRecordBatch&) { throw std::runtime_error("intentional record failure"); });

    std::mutex mutex;
    std::condition_variable cv;
    size_t good_count = 0;
    const auto good = service.register_consumer([&](const ProgramRealtimeRecordBatch& batch) {
        {
            std::lock_guard lock(mutex);
            good_count += batch.records.size();
        }
        cv.notify_all();
    });

    ring.publish(make_service_record(1, 1));
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return good_count == 1; }));
    }

    service.detach_producer(ring);
    service.unregister_consumer(throwing);
    service.unregister_consumer(good);
}

// batch.dropped counts what this consumer lost since its previous batch, not since the session began.
TEST(RealtimeProfilerSanity, DropsAreReportedSinceThePreviousCallback) {
    RealtimeProfilerService service;
    TestRecordProducer overflowing_ring(4, 4);
    TestRecordProducer blocked_ring(4, 4);
    service.attach_producer(overflowing_ring);
    service.attach_producer(blocked_ring);

    struct BatchResult {
        uint32_t chip_id;
        uint64_t dropped;
    };
    std::mutex mutex;
    std::condition_variable cv;
    bool blocked_callback_started = false;
    bool release_blocked_callback = false;
    std::vector<BatchResult> results;
    const auto handle = service.register_consumer([&](const ProgramRealtimeRecordBatch& batch) {
        std::unique_lock lock(mutex);
        const uint32_t chip_id = batch.records.front().chip_id;
        if (chip_id == 2 && !blocked_callback_started) {
            blocked_callback_started = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_blocked_callback; });
        }
        results.push_back({chip_id, batch.dropped});
        cv.notify_all();
    });

    blocked_ring.publish(make_service_record(1, 2));
    service.wake_consumers();
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return blocked_callback_started; }));
    }

    for (uint32_t id = 10; id < 22; ++id) {
        overflowing_ring.publish(make_service_record(id, 1));
    }
    service.wake_consumers();
    {
        std::lock_guard lock(mutex);
        release_blocked_callback = true;
    }
    cv.notify_all();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] {
            return std::any_of(results.begin(), results.end(), [](const BatchResult& result) {
                return result.chip_id == 1 && result.dropped > 0;
            });
        }));
    }

    service.detach_producer(overflowing_ring);
    service.detach_producer(blocked_ring);
    service.unregister_consumer(handle);
}

// detach_producer is what MeshDevice close goes through: everything already published must be delivered before it
// returns, or closing a device silently truncates its tail of records.
TEST(RealtimeProfilerSanity, DetachDrainsPublishedRecordsBeforeReturning) {
    RealtimeProfilerService service;
    TestRecordProducer ring(16, 8);
    service.attach_producer(ring);

    std::atomic<size_t> delivered = 0;
    const auto handle = service.register_consumer(
        [&](const ProgramRealtimeRecordBatch& batch) { delivered.fetch_add(batch.records.size()); });
    for (uint32_t id = 1; id <= 8; ++id) {
        ring.publish(make_service_record(id, 1));
    }
    service.wake_consumers();

    service.detach_producer(ring);
    EXPECT_EQ(delivered.load(), 8u);
    service.unregister_consumer(handle);
}

TEST(RealtimeProfilerSanity, ManyRingsAndConsumersDeliverWithoutLoss) {
    constexpr size_t kRingCount = 8;
    constexpr size_t kConsumerCount = 4;
    constexpr size_t kRecordsPerRing = 64;

    RealtimeProfilerService service;
    std::vector<std::unique_ptr<TestRecordProducer>> rings;
    rings.reserve(kRingCount);
    for (size_t i = 0; i < kRingCount; ++i) {
        rings.push_back(std::make_unique<TestRecordProducer>(128, 32));
        service.attach_producer(*rings.back());
    }

    std::vector<std::atomic<size_t>> received(kConsumerCount);
    std::vector<ProgramRealtimeProfilerCallbackHandle> handles;
    handles.reserve(kConsumerCount);
    for (size_t consumer = 0; consumer < kConsumerCount; ++consumer) {
        handles.push_back(
            service.register_consumer([&count = received[consumer]](const ProgramRealtimeRecordBatch& batch) {
                count.fetch_add(batch.records.size(), std::memory_order_relaxed);
            }));
    }

    for (size_t ring = 0; ring < kRingCount; ++ring) {
        for (size_t record = 0; record < kRecordsPerRing; ++record) {
            rings[ring]->publish(make_service_record(
                static_cast<uint32_t>(ring * kRecordsPerRing + record), static_cast<uint32_t>(ring)));
        }
    }
    service.wake_consumers();

    const auto deadline = std::chrono::steady_clock::now() + 5s;
    const auto all_delivered = [&] {
        return std::all_of(received.begin(), received.end(), [](const std::atomic<size_t>& count) {
            return count.load(std::memory_order_relaxed) == kRingCount * kRecordsPerRing;
        });
    };
    while (std::chrono::steady_clock::now() < deadline && !all_delivered()) {
        std::this_thread::sleep_for(1ms);
    }
    for (const auto& count : received) {
        EXPECT_EQ(count.load(std::memory_order_relaxed), kRingCount * kRecordsPerRing);
    }

    for (auto& ring : rings) {
        service.detach_producer(*ring);
    }
    for (auto handle : handles) {
        service.unregister_consumer(handle);
    }
}

TEST_F(RealtimeProfilerDeviceSanity, RecordsAreWellFormedAndCarryTheirProgramsSources) {
    RecordCollector collector;
    enqueue_programs(kNumPrograms);
    quiesce_and_wait_for_programs(collector, kNumPrograms);
    collector.stop();

    const std::vector<ProgramRealtimeRecord> records = collector.records();
    ASSERT_GE(records.size(), kNumPrograms)
        << "Expected at least " << kNumPrograms << " RT profiler records (one per program), got " << records.size();
    EXPECT_EQ(collector.dropped(), 0u);

    for (const auto& rec : records) {
        EXPECT_GT(rec.end_timestamp, rec.start_timestamp)
            << "RT record end_timestamp must be strictly greater than start_timestamp (runtime_id=" << rec.runtime_id
            << ", chip=" << rec.chip_id << ")";
        EXPECT_GT(rec.frequency, 0.0) << "RT record frequency must be positive (runtime_id=" << rec.runtime_id
                                      << ", chip=" << rec.chip_id << ")";

        EXPECT_GT(rec.clock_sync.sync_error, std::chrono::nanoseconds::zero())
            << "RT record sync_error should be populated once the clock is anchored (runtime_id=" << rec.runtime_id
            << ")";

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
}

TEST_F(RealtimeProfilerDeviceSanity, CloseDrainsRegisteredCallback) {
    RecordCollector collector;
    enqueue_programs(kNumPrograms);

    mesh_device_->quiesce_devices();
    EXPECT_TRUE(close_mesh());
    collector.stop();

    EXPECT_EQ(collector.runtime_ids_in_range(kNumPrograms).size(), kNumPrograms)
        << "Mesh close should drain records for callbacks still registered at shutdown";
}

// Finish, unlike close, leaves the device open, so the final program's record only arrives if a finish-time flush
// is emitted rather than waiting for the next program to push it out.
TEST_F(RealtimeProfilerDeviceSanity, LastProgramRecordDeliveredOnFinish) {
    RecordCollector collector;
    enqueue_programs(kNumPrograms);

    distributed::Finish(mesh_device_->mesh_command_queue());
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!collector.runtime_ids_in_range(kNumPrograms).contains(kNumPrograms) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
    collector.stop();

    EXPECT_TRUE(collector.runtime_ids_in_range(kNumPrograms).contains(kNumPrograms))
        << "the final program's record (runtime_id=" << kNumPrograms << ") was not delivered";
}

TEST_F(RealtimeProfilerDeviceSanityWithTrace, TraceReplayResolvesKernelSources) {
    constexpr uint32_t kWarmupRuntimeId = 0x6001;
    constexpr uint32_t kTraceRuntimeId = 0x6002;

    RecordCollector collector;
    const CoreRange cores = all_cores();
    Program program =
        make_sanity_program(make_sanity_kernel_source(kTraceRuntimeId), cores, /*runtime_id=*/kWarmupRuntimeId);

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device_->shape()), std::move(program));
    auto& mesh_cq = mesh_device_->mesh_command_queue(0);

    // Warm up before capture (capture cannot load binaries) under kWarmupRuntimeId, then switch to
    // kTraceRuntimeId so the trace-baked id is tied only by create_trace_node, the path under test.
    distributed::EnqueueMeshWorkload(mesh_cq, workload, true);
    for (auto& [_, prog] : workload.get_programs()) {
        prog.set_runtime_id(static_cast<uint64_t>(kTraceRuntimeId));
    }

    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device_.get(), mesh_cq.id());
    distributed::EnqueueMeshWorkload(mesh_cq, workload, false);
    mesh_device_->end_mesh_trace(mesh_cq.id(), trace_id);
    mesh_device_->replay_mesh_trace(mesh_cq.id(), trace_id, true);

    quiesce_and_wait_for([&] {
        return std::ranges::any_of(collector.records(), [](const ProgramRealtimeRecord& record) {
            return record.runtime_id == kTraceRuntimeId;
        });
    });
    collector.stop();
    mesh_device_->release_mesh_trace(trace_id);

    const std::string expected_marker = kSourceMarkerPrefix + std::to_string(kTraceRuntimeId);
    uint32_t trace_records = 0;
    for (const auto& rec : collector.records()) {
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
}

// Independent of the Tracy SYNC_CHECK markers, whose host and device endpoints are both derived from the same
// calibration and so coincide by construction: they cannot reveal a wrong mapping. This test instead brackets each
// program's on-device execution between two host-clock reads and asserts the record's device->host mapping --
// reconstructed solely from its own device_cycle_offset and frequency -- lands inside that independently-measured
// host window.
TEST_F(RealtimeProfilerDeviceSanity, RecordHostTimeFallsInDispatchWindow) {
    RecordCollector collector;
    const CoreRange cores = all_cores();

    // One fixed kernel source for every program so the JIT ELF is compiled once (during warm-up) and reused, keeping
    // each measured dispatch window free of kernel-compilation time.
    const std::string fixed_src = make_sanity_kernel_source(/*runtime_id=*/0);

    auto enqueue_blocking = [&](uint32_t runtime_id) {
        enqueue_rt_program(mesh_device_, make_sanity_program(fixed_src, cores, runtime_id), /*blocking=*/true);
    };

    enqueue_blocking(/*runtime_id=*/1);  // JIT warm-up; not placed in the window map

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

    quiesce_and_wait_for([&] { return collector.records().size() >= kNumMeasured; });
    collector.stop();

    // Covers host-clock read jitter and the gap between the bracketing reads and actual dispatch/completion. Catches
    // a fundamentally wrong mapping (off by ms and up), not µs-level skew.
    constexpr auto kSlack = std::chrono::milliseconds(2);

    int checked = 0;
    std::chrono::nanoseconds worst_outside{};
    double min_freq = 0.0;
    double max_freq = 0.0;
    for (const auto& rec : collector.records()) {
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
}

}  // namespace
}  // namespace tt::tt_metal
