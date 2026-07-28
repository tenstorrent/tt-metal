// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merge-gate sanity coverage for the real-time (RT) profiler, in three layers:
//
//   RealtimeProfilerClockModel / RealtimeProfilerRttFloor -- the host-side clock logic, exercised directly with
//       synthetic handshakes. No device, so these also run wherever the profiler itself cannot.
//   RealtimeProfilerSanity -- the record service and its consumer threads, driven through hand-fed rings. A MeshDevice
//       contributes exactly one ring, so multi-ring and mid-callback behaviour is unreachable from a device test.
//   RealtimeProfilerDeviceSanity -- the whole pipeline on a unit mesh: mailbox layout, D2H socket init, sync
//       handshake, kernel source propagation, timestamp extraction.
//
// Lives in the dispatch "basic" test library so it runs as part of `tt-metalium-validation-basic`, which the merge-gate
// `metalium-basic-tests` job executes on both N150 (WH) and P150b (BH). The device layer skips gracefully via
// IsProgramRealtimeProfilerActive() where the profiler cannot be enabled (ETH dispatch, non-MMIO chip, kernels
// nullified, IOMMU-off on BH).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <iostream>
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

#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "impl/context/metal_context.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/realtime_profiler/realtime_profiler_clock_model.hpp"
#include "impl/realtime_profiler/realtime_profiler_receiver.hpp"
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

// clock_sync.sync_error_ns is half the sync-handshake RTT: the minimax bound on where the anchor landed, and what each
// record self-reports as its accuracy. Asserted as a distribution rather than a per-record ceiling because the two
// failure modes look different -- a degraded handshake lifts the median, while a re-anchor policy that stops rejecting
// bad round trips only fattens the tail.
//
// Sized for the slower of the two architectures. Blackhole reaches the profiler core through a TLB window and reports
// ~0.6us; Wormhole goes through write_core_immediate and a poll for a round trip of ~4-5us, so it reports ~2us p50 and
// ~12us p99. Tightening these to what Blackhole does would fail every Wormhole run.
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
    // The subset of 1..count that produced a record, which is what most device tests actually assert on.
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

// Opens a unit mesh with the RT profiler active, or skips. Activation is decided during the init-sync handshake inside
// mesh open, so the check is stable by the time create_unit_mesh returns.
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
        if (!IsProgramRealtimeProfilerActive()) {
            close_mesh();
            GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
        }
    }

    void TearDown() override { close_mesh(); }

    // Only TraceReplayResolvesKernelSources needs a trace region; everything else pays nothing for it.
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

    // Runtime IDs start at 1 so every program emits a record; runtime_id == 0 is reserved for infrastructure traffic
    // and filtered host-side.
    void enqueue_programs(uint32_t count) {
        for (uint32_t i = 1; i <= count; ++i) {
            enqueue_sanity_program(mesh_device_, i, all_cores());
        }
    }

    // Waits out the receiver thread's last drain. 500ms mirrors the programming example at
    // test_realtime_profiler_csv.cpp and has proven sufficient for small workloads on WH/BH single-chip.
    void quiesce_and_settle() {
        mesh_device_->quiesce_devices();
        std::this_thread::sleep_for(500ms);
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
};

class RealtimeProfilerDeviceSanityWithTrace : public RealtimeProfilerDeviceSanity {
protected:
    size_t trace_region_size() const override { return 8 * 1024 * 1024; }
};

// Synthetic steady_clock instants. Real ones are ~1e15 ns since boot, close enough to the 2^53 limit of exact integer
// arithmetic in a double that the model's own rounding would show up in the assertions below.
constexpr std::chrono::steady_clock::time_point host_instant(int64_t ns) {
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}

// Handshakes lying exactly on device_ticks = frequency * host_ns + ticks_at_start, spaced like the bring-up fit.
std::vector<ClockSyncSample> make_fit_samples(
    std::chrono::steady_clock::time_point start, double frequency, uint64_t ticks_at_start, size_t count) {
    std::vector<ClockSyncSample> samples;
    samples.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        const auto host_time = start + std::chrono::milliseconds(5 * static_cast<int64_t>(i));
        const double elapsed_ns = static_cast<double>((host_time - start).count());
        samples.push_back(ClockSyncSample{
            .host_time = host_time,
            .rtt = std::chrono::nanoseconds(1200),
            .device_ticks = ticks_at_start + static_cast<uint64_t>(frequency * elapsed_ns)});
    }
    return samples;
}

TEST(RealtimeProfilerClockModel, FitRecoversTheDeviceClockFrequency) {
    ClockModel model;
    model.seed_frequency(1.0);
    const auto start = host_instant(1'000'000'000);
    const auto samples = make_fit_samples(start, /*frequency=*/1.35, /*ticks_at_start=*/1'000'000, /*count=*/100);

    const ClockFitQuality quality = model.fit(samples, start);

    EXPECT_TRUE(quality.ok);
    EXPECT_EQ(quality.num_samples, 100u);
    EXPECT_NEAR(model.frequency(), 1.35, 1e-9);
    EXPECT_TRUE(model.is_anchored());
    // Samples sit on the line up to the truncation of device_ticks to whole cycles, i.e. under one cycle of residual.
    EXPECT_LT(quality.residual_rms_ns, 1.0);
}

TEST(RealtimeProfilerClockModel, TooFewSamplesKeepTheSeededFrequency) {
    ClockModel model;
    model.seed_frequency(1.35);
    const auto start = host_instant(1'000'000'000);

    const ClockFitQuality quality = model.fit(make_fit_samples(start, /*frequency=*/0.5, 0, /*count=*/1), start);

    EXPECT_FALSE(quality.ok);
    EXPECT_EQ(model.frequency(), 1.35);
    EXPECT_FALSE(model.is_anchored()) << "a mapping with no handshake behind it must not look anchored";
}

TEST(RealtimeProfilerClockModel, NonPositiveFittedSlopeKeepsTheSeededFrequency) {
    // Consumers divide by the frequency, so a device whose tick count appears to run backwards must not reach them.
    ClockModel model;
    model.seed_frequency(1.35);
    const auto start = host_instant(1'000'000'000);
    std::vector<ClockSyncSample> samples;
    for (size_t i = 0; i < 8; ++i) {
        samples.push_back(ClockSyncSample{
            .host_time = start + std::chrono::milliseconds(5 * static_cast<int64_t>(i)),
            .rtt = std::chrono::nanoseconds(1200),
            .device_ticks = 100'000 - i * 1'000});
    }

    model.fit(samples, start);

    EXPECT_EQ(model.frequency(), 1.35);
}

TEST(RealtimeProfilerClockModel, AnchorSitsAtTheRoundTripMidpoint) {
    ClockModel model;
    model.seed_frequency(1.0);  // one tick per nanosecond, so the mapping arithmetic below is exact
    const auto now = host_instant(1'000'000'000);
    constexpr auto kRtt = std::chrono::nanoseconds(1000);
    const ClockSyncSample sample{.host_time = now, .rtt = kRtt, .device_ticks = 500'000};

    model.reanchor(now, sample);

    const auto mapping = model.mapping(now);
    const int64_t mapped_host_ns = static_cast<int64_t>(
        (static_cast<double>(sample.device_ticks) - static_cast<double>(mapping.device_cycle_offset)) /
        model.frequency());
    EXPECT_EQ(mapped_host_ns, (now + kRtt / 2).time_since_epoch().count())
        << "the device timestamp must map back to the midpoint of the round trip that carried it";
    EXPECT_EQ(mapping.sync_error_ns, static_cast<uint64_t>((kRtt / 2).count()));
}

TEST(RealtimeProfilerClockModel, FirstHandshakeIsAcceptedHoweverSlow) {
    ClockModel model;
    model.seed_frequency(1.0);
    EXPECT_TRUE(model.accept_reanchor(std::chrono::seconds(1), host_instant(0)));
}

TEST(RealtimeProfilerClockModel, SlowHandshakeIsRejectedWhileTheAnchorIsFresh) {
    ClockModel model;
    model.seed_frequency(1.0);
    const auto anchored_at = host_instant(1'000'000'000);
    model.reanchor(anchored_at, ClockSyncSample{.host_time = anchored_at, .rtt = std::chrono::nanoseconds(1200)});

    const auto soon_after = anchored_at + std::chrono::milliseconds(10);
    EXPECT_TRUE(model.accept_reanchor(std::chrono::microseconds(2), soon_after));
    EXPECT_FALSE(model.accept_reanchor(std::chrono::milliseconds(1), soon_after))
        << "a round trip this slow places the anchor worse than the drift it would correct";
}

TEST(RealtimeProfilerClockModel, StaleAnchorAcceptsEvenASlowHandshake) {
    ClockModel model;
    model.seed_frequency(1.0);
    const auto anchored_at = host_instant(1'000'000'000);
    model.reanchor(anchored_at, ClockSyncSample{.host_time = anchored_at, .rtt = std::chrono::nanoseconds(1200)});

    EXPECT_FALSE(model.accept_reanchor(std::chrono::milliseconds(1), anchored_at + std::chrono::seconds(1)));
    EXPECT_TRUE(model.accept_reanchor(std::chrono::milliseconds(1), anchored_at + std::chrono::seconds(5)))
        << "once the anchor has gone stale, a loose anchor beats unbounded drift";
}

TEST(RealtimeProfilerRttFloor, HasNoTargetBeforeTheFirstObservation) {
    const RttFloor floor;
    EXPECT_EQ(floor.value(), std::chrono::nanoseconds::zero());
    EXPECT_FALSE(floor.is_near(std::chrono::nanoseconds(1)));
}

TEST(RealtimeProfilerRttFloor, DropsToANewMinimumImmediately) {
    RttFloor floor;
    floor.observe(std::chrono::nanoseconds(2000));
    EXPECT_EQ(floor.value(), std::chrono::nanoseconds(2000));
    floor.observe(std::chrono::nanoseconds(1200));
    EXPECT_EQ(floor.value(), std::chrono::nanoseconds(1200));
}

TEST(RealtimeProfilerRttFloor, TreatsASmallExcessOverTheFloorAsNearIt) {
    RttFloor floor;
    floor.observe(std::chrono::nanoseconds(1000));
    EXPECT_TRUE(floor.is_near(std::chrono::nanoseconds(1040)));
    EXPECT_FALSE(floor.is_near(std::chrono::nanoseconds(1500)));
}

TEST(RealtimeProfilerRttFloor, RisesWhenTheStepIsBelowOneNanosecond) {
    // At a realistic floor the per-observation rise is a fraction of a nanosecond. Truncating it away would freeze the
    // floor at the all-time minimum, and a burst would never exit early again.
    RttFloor floor;
    floor.observe(std::chrono::nanoseconds(1200));
    for (int i = 0; i < 10; ++i) {
        floor.observe(std::chrono::nanoseconds(5000));
    }
    EXPECT_GT(floor.value(), std::chrono::nanoseconds(1200));
}

TEST(RealtimeProfilerRttFloor, SustainedSlowHandshakesDoNotDragTheFloorToTheirMean) {
    // A floor that rose in proportion to each sample's excess would converge on the distribution's mean and then admit
    // everything, which is the one thing the burst's early exit must not do.
    RttFloor floor;
    floor.observe(std::chrono::nanoseconds(1200));
    for (int i = 0; i < 500; ++i) {
        floor.observe(std::chrono::nanoseconds(4000));
    }
    EXPECT_LT(floor.value(), std::chrono::nanoseconds(1600));
    EXPECT_FALSE(floor.is_near(std::chrono::nanoseconds(4000)));
}

// The service tests drive RealtimeProfilerService directly rather than opening a mesh: a MeshDevice contributes
// exactly one record ring, so multi-ring delivery, drop accounting and mid-callback control changes are all
// unreachable from a device test. The ring itself is covered by tests/tt_metal/tt_metal/misc/test_broadcast_ring.cpp.
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

// A reader starts at the ring's current head, so a consumer registering late sees the stream from that point on rather
// than replaying whatever backlog happens to still be resident.
TEST(RealtimeProfilerSanity, LateConsumerReceivesOnlyFutureRecords) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(8);
    service.attach_ring(ring, 4);

    ring.writer().publish(make_service_record(1, 7));
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

    ring.writer().publish(make_service_record(2, 7));
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return !received.empty(); }));
        EXPECT_EQ(received, (std::vector<uint32_t>{2}));
    }

    service.detach_ring(ring);
    service.unregister_consumer(handle);
}

// Consumers are isolated from each other: one that throws every time must not cost a sibling any records. The
// device-level ThrowingCallbackIsIsolated covers the same contract end-to-end, but only this one pins it to the
// service without a mesh in the way.
TEST(RealtimeProfilerSanity, ThrowingConsumerDoesNotAffectSibling) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(8);
    service.attach_ring(ring, 4);

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

    ring.writer().publish(make_service_record(1, 1));
    service.wake_consumers();

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return good_count == 1; }));
    }

    service.detach_ring(ring);
    service.unregister_consumer(throwing);
    service.unregister_consumer(good);
}

// batch.dropped counts what this consumer lost since its previous batch, not since the session began. A consumer stuck
// in a callback overruns the ring it is not reading, and the loss has to surface on its next batch from that ring.
TEST(RealtimeProfilerSanity, DropsAreReportedSinceThePreviousCallback) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing overflowing_ring(4);
    RealtimeProfilerRecordRing blocked_ring(4);
    service.attach_ring(overflowing_ring, 4);
    service.attach_ring(blocked_ring, 4);

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

    blocked_ring.writer().publish(make_service_record(1, 2));
    service.wake_consumers();
    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, 5s, [&] { return blocked_callback_started; }));
    }

    for (uint32_t id = 10; id < 22; ++id) {
        overflowing_ring.writer().publish(make_service_record(id, 1));
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

    service.detach_ring(overflowing_ring);
    service.detach_ring(blocked_ring);
    service.unregister_consumer(handle);
}

// detach_ring is what MeshDevice close goes through, so everything already published has to be delivered before it
// returns; otherwise closing a device silently truncates its tail of records.
TEST(RealtimeProfilerSanity, DetachDrainsPublishedRecordsBeforeReturning) {
    RealtimeProfilerService service;
    RealtimeProfilerRecordRing ring(16);
    service.attach_ring(ring, 8);

    std::atomic<size_t> delivered = 0;
    const auto handle = service.register_consumer(
        [&](const ProgramRealtimeRecordBatch& batch) { delivered.fetch_add(batch.records.size()); });
    for (uint32_t id = 1; id <= 8; ++id) {
        ring.writer().publish(make_service_record(id, 1));
    }
    service.wake_consumers();

    service.detach_ring(ring);
    EXPECT_EQ(delivered.load(), 8u);
    service.unregister_consumer(handle);
}

// A consumer thread parks on the wake generation when there is nothing to read; destruction has to wake it or the
// destructor blocks forever joining it.
TEST(RealtimeProfilerSanity, DestructorWakesIdleConsumerThread) {
    RealtimeProfilerService service;
    service.register_consumer([](const ProgramRealtimeRecordBatch&) {});
}

TEST(RealtimeProfilerSanity, ManyRingsAndConsumersDeliverWithoutLoss) {
    constexpr size_t kRingCount = 8;
    constexpr size_t kConsumerCount = 4;
    constexpr size_t kRecordsPerRing = 64;

    RealtimeProfilerService service;
    std::vector<std::unique_ptr<RealtimeProfilerRecordRing>> rings;
    rings.reserve(kRingCount);
    for (size_t i = 0; i < kRingCount; ++i) {
        rings.push_back(std::make_unique<RealtimeProfilerRecordRing>(128));
        service.attach_ring(*rings.back(), 32);
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
            rings[ring]->writer().publish(make_service_record(
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
        service.detach_ring(*ring);
    }
    for (auto handle : handles) {
        service.unregister_consumer(handle);
    }
}

TEST_F(RealtimeProfilerDeviceSanity, FiveProgramsBackToBack) {
    RecordCollector collector;
    enqueue_programs(kNumPrograms);
    quiesce_and_settle();
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
TEST_F(RealtimeProfilerDeviceSanity, SyncAccuracy) {
    RecordCollector collector;
    // A fixed runtime_id keeps the kernel source (and its JIT compile) shared across iterations; the 50ms spacing
    // straddles the resync interval so successive records fall in different re-anchor epochs.
    constexpr uint32_t kIterations = 40;
    for (uint32_t i = 0; i < kIterations; ++i) {
        enqueue_sanity_program(mesh_device_, /*runtime_id=*/1, all_cores());
        std::this_thread::sleep_for(50ms);
    }
    quiesce_and_settle();
    collector.stop();

    std::vector<uint64_t> sync_errors_ns;
    for (const auto& record : collector.records()) {
        sync_errors_ns.push_back(record.clock_sync.sync_error_ns);
    }
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

TEST_F(RealtimeProfilerDeviceSanity, ThrowingCallbackIsIsolated) {
    std::atomic<uint64_t> throwing_invocations = 0;
    const auto throwing_handle =
        RegisterProgramRealtimeProfilerCallback([&throwing_invocations](const ProgramRealtimeRecordBatch&) {
            throwing_invocations.fetch_add(1);
            throw std::runtime_error("intentional callback failure");
        });
    RecordCollector collector;

    enqueue_programs(kNumPrograms);
    quiesce_and_settle();
    UnregisterProgramRealtimeProfilerCallback(throwing_handle);
    collector.stop();

    EXPECT_GT(throwing_invocations.load(), 0u) << "throwing callback should have been invoked";
    EXPECT_EQ(collector.runtime_ids_in_range(kNumPrograms).size(), kNumPrograms)
        << "sibling callback must receive every record despite the other callback throwing";
}

// Finish, unlike close, leaves the device open, so the final program's record only arrives if the finish-time flush
// is emitted rather than waiting for the next program to push it out.
TEST_F(RealtimeProfilerDeviceSanity, LastProgramRecordDeliveredOnFinish) {
    RecordCollector collector;
    enqueue_programs(kNumPrograms);

    distributed::Finish(mesh_device_->mesh_command_queue());
    std::this_thread::sleep_for(500ms);
    collector.stop();

    EXPECT_TRUE(collector.runtime_ids_in_range(kNumPrograms).contains(kNumPrograms))
        << "the final program's record (runtime_id=" << kNumPrograms << ") was not delivered";
}

TEST_F(RealtimeProfilerDeviceSanityWithTrace, TraceReplayResolvesKernelSources) {
    constexpr uint32_t kWarmupRuntimeId = 0x6001;
    constexpr uint32_t kTraceRuntimeId = 0x6002;

    RecordCollector collector;
    const CoreRange cores = all_cores();
    const std::string kernel_src = make_sanity_kernel_source(kTraceRuntimeId);
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
    program.set_runtime_id(static_cast<uint64_t>(kWarmupRuntimeId));

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

    quiesce_and_settle();
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

// Independent host/device sync-accuracy check. The Tracy SYNC_CHECK markers are a self-consistency check — their host
// and device endpoints are both derived from the same calibration, so they coincide by construction and cannot reveal
// a wrong mapping. This test instead brackets each program's on-device execution between two host-clock reads and
// asserts the record's device->host mapping — reconstructed solely from the record's own device_cycle_offset and
// frequency — lands inside that independently-measured host window. A mis-signed/mis-scaled offset, a wrong clock
// domain, or a stale anchor would push the record outside the window. This is only possible because the affine mapping
// now rides on the record itself.
TEST_F(RealtimeProfilerDeviceSanity, RecordHostTimeFallsInDispatchWindow) {
    RecordCollector collector;
    const CoreRange cores = all_cores();

    // One fixed kernel source for every program so the JIT ELF is compiled once (during warm-up) and reused; this
    // keeps each measured dispatch window free of multi-second kernel-compilation time.
    const std::string fixed_src = make_sanity_kernel_source(/*runtime_id=*/0);

    auto enqueue_blocking = [&](uint32_t runtime_id) {
        Program program = CreateProgram();
        CreateKernelFromString(
            program,
            fixed_src,
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        CreateKernelFromString(
            program,
            fixed_src,
            cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        CreateKernelFromString(program, fixed_src, cores, ComputeConfig{});
        program.set_runtime_id(static_cast<uint64_t>(runtime_id));
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device_->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, /*blocking=*/true);
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

    quiesce_and_settle();
    collector.stop();

    // Generous bound: it covers host-clock read jitter and the gap between the bracketing reads and the actual
    // dispatch/completion. The check catches a fundamentally wrong mapping (off by ms and up), not µs-level skew.
    constexpr double kSlackNs = 2'000'000.0;

    int checked = 0;
    double worst_outside_ns = 0.0;
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
}

// Devices take turns being resynced, one per tick, rather than all together on a single tick. A rotation that failed
// to advance would leave every device but the first pinned to its bring-up fit, drifting with nothing to correct it,
// while the published sync_error_ns still looked healthy -- so the counts are the only thing that shows it.
TEST(RealtimeProfilerSanity, EveryDeviceIsResyncedInRotation) {
    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    auto* receiver = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(receiver, nullptr);
    // A full rotation takes the resync interval; sleep well past several of them.
    std::this_thread::sleep_for(1s);
    const std::vector<uint64_t> resyncs = receiver->num_resyncs_per_device();

    ASSERT_EQ(resyncs.size(), receiver->num_active_devices());
    ASSERT_FALSE(resyncs.empty());
    const auto [min_it, max_it] = std::minmax_element(resyncs.begin(), resyncs.end());
    EXPECT_GT(*min_it, 0u) << "at least one device was never resynced; the rotation is not advancing";
    // Strict round-robin, so the spread is one visit plus whatever the sample raced with.
    EXPECT_LE(*max_it - *min_it, 2u) << "resyncs are not evenly distributed across devices";

    EXPECT_TRUE(mesh_device->close());
}

// Independent check of the clock mapping the RT profiler publishes on every record. It shares no code and no
// mechanism with the production sync: production has the host drive a round trip and bracket a device timestamp
// inside it, while this reads the device's own clock and brackets that read between two host clock reads.
//
// Nothing is fitted here. A record already states, through device_cycle_offset and frequency, which host time a
// device timestamp corresponds to, so there is no line to reconstruct -- only its residual to measure. One read
// resolves nothing on its own, because the host bracket is wider than the error being looked for; averaging many is
// what gives the measurement its resolution. The reads are taken back-to-back, so a round spans milliseconds and no
// clock drifts across it, and the mean residual is good to about one bracket over the square root of the sample
// count -- tens of nanoseconds, against roughly a microsecond for any single read.
//
// The kernel never waits on the host. It stamps WALL_CLOCK into one L1 slot in a bounded loop and exits, so the host
// can stop reading whenever it likes -- or fail an assertion and leave -- without stranding a spinning kernel, which
// would make Finish, and the test, hang rather than fail.
constexpr uint32_t kClockStampIterations = 200'000'000;  // ~5s of stamping; only reached if the stop flag never lands
constexpr uint32_t kClockCheckRounds = 3;
// Taken back-to-back, a couple of microseconds apart, so a round's reads span only milliseconds. Enough of them that
// the mean residual resolves well below a single bracket.
constexpr uint32_t kClockReadsPerRound = 500;
constexpr uint32_t kFirstClockCheckRuntimeId = 9101;
// What this has to establish: that the published mapping is right to within a few microseconds. Deliberately fixed
// rather than derived from the record's own sync_error_ns -- a bound computed from the claim cannot test the claim.
// The measured disagreement is ~210ns, so these leave an order of magnitude of headroom, and they are wide enough to
// absorb the clock drift between a record's mapping snapshot and the reads it is checked against.
constexpr double kMaxMeanMappingErrorNs = 2'000.0;
constexpr double kMaxTailMappingErrorNs = 3'000.0;

std::string make_clock_stamp_kernel_source() {
    return R"(
#include <cstdint>
#include "risc_common.h"

void kernel_main() {
    const uint32_t stamp_addr = get_arg_val<uint32_t>(0);
    const uint32_t stop_addr = get_arg_val<uint32_t>(1);
    const uint32_t max_iterations = get_arg_val<uint32_t>(2);

    volatile tt_l1_ptr uint32_t* stamp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stamp_addr);
    volatile tt_l1_ptr const uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);

    for (uint32_t i = 0; i < max_iterations; i++) {
        const uint64_t now = get_timestamp();
        stamp[1] = static_cast<uint32_t>(now >> 32);
        stamp[0] = static_cast<uint32_t>(now & 0xFFFFFFFF);
        if ((i & 0xFF) == 0 && stop[0] != 0) {
            break;
        }
    }
}
)";
}

// One read of the device clock, bracketed by the host's own clock either side of it.
struct ClockBracket {
    int64_t host_before_ns = 0;
    int64_t host_after_ns = 0;
    uint64_t device_ticks = 0;
    uint32_t round = 0;

    double host_mid_ns() const {
        return static_cast<double>(host_before_ns) + static_cast<double>(host_after_ns - host_before_ns) / 2.0;
    }
    double half_width_ns() const { return static_cast<double>(host_after_ns - host_before_ns) / 2.0; }
};

TEST_F(RealtimeProfilerDeviceSanity, RecordMappingMatchesAnIndependentClockRead) {
    RecordCollector collector;

    IDevice* device = mesh_device_->get_devices().front();
    const uint32_t l1_base = mesh_device_->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t stamp_addr = l1_base;
    const uint32_t stop_addr = l1_base + 16;
    const CoreCoord core{0, 0};
    const CoreCoord vcore = device->virtual_core_from_logical_core(core, CoreType::WORKER);
    auto& cluster = MetalContext::instance().get_cluster();

    // Blackhole exposes a static TLB window onto the core's L1, so the device clock can be read with one load instead
    // of going through UMD's generic path. Only software overhead comes off -- an MMIO read is a non-posted PCIe
    // transaction either way -- but it is the widest part of the bracket that can be removed at all. Null elsewhere,
    // where the generic path is used instead.
    volatile uint64_t* mapped_stamp = nullptr;
    try {
        // is_tlb_mapped asks the question that actually matters -- are these bytes covered by a mapping that will not
        // move underneath a raw pointer -- rather than inferring it from the architecture.
        const tt_xy_pair tlb_core(vcore.x, vcore.y);
        auto* tlb_manager = cluster.get_driver()->get_chip(device->id())->get_tlb_manager();
        if (tlb_manager != nullptr && tlb_manager->is_tlb_mapped(tlb_core, stamp_addr, sizeof(uint64_t))) {
            // A lookup, never an allocation: get_tlb_window throws unless UMD already mapped this core at init, and
            // is_tlb_mapped has just confirmed the mapping covers these bytes. No TLB is consumed by reading here.
            auto* window = tlb_manager->get_tlb_window(tlb_core);
            if (window != nullptr) {
                const uint64_t window_offset =
                    window->get_base_address() - window->handle_ref().get_config().local_offset;
                mapped_stamp =
                    reinterpret_cast<volatile uint64_t*>(window->handle_ref().get_base() + stamp_addr + window_offset);
            }
        }
    } catch (const std::exception&) {
        // Left null; the generic read path still works.
    }

    const auto host_now_ns = [] { return std::chrono::steady_clock::now().time_since_epoch().count(); };
    const auto read_device_ticks = [&]() -> uint64_t {
        if (mapped_stamp != nullptr) {
            return *mapped_stamp;
        }
        uint32_t words[2] = {0, 0};
        cluster.read_core(words, sizeof(words), tt_cxy_pair(device->id(), vcore), stamp_addr);
        return (static_cast<uint64_t>(words[1]) << 32) | words[0];
    };

    // Each round brackets one device clock read, then runs one program so a record is published with the mapping that
    // was live beside that read. Pairing them keeps every comparison local in time.
    std::vector<ClockBracket> brackets;
    for (uint32_t round = 0; round < kClockCheckRounds; ++round) {
        const std::vector<uint32_t> zeros(8, 0);
        cluster.write_core(zeros.data(), zeros.size() * sizeof(uint32_t), tt_cxy_pair(device->id(), vcore), stamp_addr);

        Program stamp_program = CreateProgram();
        auto kernel = CreateKernelFromString(
            stamp_program,
            make_clock_stamp_kernel_source(),
            CoreRangeSet(CoreRange(core, core)),
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(stamp_program, kernel, core, {stamp_addr, stop_addr, kClockStampIterations});
        distributed::MeshWorkload stamp_workload;
        stamp_workload.add_program(distributed::MeshCoordinateRange(mesh_device_->shape()), std::move(stamp_program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), stamp_workload, /*blocking=*/false);

        // Wait for the slot to go non-zero rather than assume a launch latency. A timeout fails the assertion; it
        // cannot hang, because the kernel ends itself either way.
        const auto launch_deadline = std::chrono::steady_clock::now() + 10s;
        while (read_device_ticks() == 0 && std::chrono::steady_clock::now() < launch_deadline) {
            std::this_thread::sleep_for(1ms);
        }
        ASSERT_NE(read_device_ticks(), 0u) << "the stamping kernel never started";

        // The window arithmetic is only exercised on the architectures this has actually run on, so prove the pointer
        // before trusting a clock read through it. The device clock only rises, so a mapped read taken between two
        // generic reads has to fall between them; anything else means the pointer does not address what it is
        // believed to, and the generic path carries the rest of the test.
        if (mapped_stamp != nullptr && round == 0) {
            const auto generic_read = [&] {
                uint32_t words[2] = {0, 0};
                cluster.read_core(words, sizeof(words), tt_cxy_pair(device->id(), vcore), stamp_addr);
                return (static_cast<uint64_t>(words[1]) << 32) | words[0];
            };
            const uint64_t before = generic_read();
            const uint64_t mapped = *mapped_stamp;
            const uint64_t after = generic_read();
            if (mapped < before || mapped > after) {
                ADD_FAILURE() << "the mapped clock read (" << mapped << ") fell outside two generic reads bracketing "
                              << "it (" << before << ", " << after << "); falling back to the generic path";
                mapped_stamp = nullptr;
            }
        }

        for (uint32_t i = 0; i < kClockReadsPerRound; ++i) {
            ClockBracket bracket;
            bracket.host_before_ns = host_now_ns();
            bracket.device_ticks = read_device_ticks();
            bracket.host_after_ns = host_now_ns();
            bracket.round = round;
            brackets.push_back(bracket);
        }

        const uint32_t stop = 1;
        cluster.write_core_immediate(&stop, sizeof(stop), tt_cxy_pair(device->id(), vcore), stop_addr);
        distributed::Finish(mesh_device_->mesh_command_queue());

        enqueue_sanity_program(mesh_device_, kFirstClockCheckRuntimeId + round, all_cores());
        distributed::Finish(mesh_device_->mesh_command_queue());
    }

    quiesce_and_settle();
    collector.stop();

    std::map<uint32_t, ProgramRealtimeRecord> record_by_runtime_id;
    for (const auto& record : collector.records()) {
        if (record.frequency > 0.0) {
            record_by_runtime_id.insert({record.runtime_id, record});
        }
    }

    // Residual of the published mapping against each independently bracketed read: where the mapping says the device
    // timestamp happened, minus where the host observed it happening.
    std::vector<double> residuals_ns;
    residuals_ns.reserve(brackets.size());
    std::vector<double> bracket_widths_ns;
    uint64_t claimed_error_ns = 0;
    uint32_t rounds_checked = 0;
    for (uint32_t round = 0; round < kClockCheckRounds; ++round) {
        const auto it = record_by_runtime_id.find(kFirstClockCheckRuntimeId + round);
        if (it == record_by_runtime_id.end()) {
            continue;
        }
        ++rounds_checked;
        const ProgramRealtimeRecord& record = it->second;
        for (const auto& bracket : brackets) {
            if (bracket.round != round) {
                continue;
            }
            const double mapped_host_ns = (static_cast<double>(bracket.device_ticks) -
                                           static_cast<double>(record.clock_sync.device_cycle_offset)) /
                                          record.frequency;
            residuals_ns.push_back(mapped_host_ns - bracket.host_mid_ns());
            bracket_widths_ns.push_back(bracket.half_width_ns());
            claimed_error_ns = std::max(claimed_error_ns, record.clock_sync.sync_error_ns);
        }
    }
    ASSERT_GE(rounds_checked, kClockCheckRounds / 2) << "too few rounds produced a record to check against";
    ASSERT_FALSE(residuals_ns.empty());

    const double mean_residual_ns =
        std::accumulate(residuals_ns.begin(), residuals_ns.end(), 0.0) / static_cast<double>(residuals_ns.size());
    std::vector<double> magnitudes;
    magnitudes.reserve(residuals_ns.size());
    for (const double residual : residuals_ns) {
        magnitudes.push_back(std::abs(residual));
    }
    std::ranges::sort(magnitudes);
    // The median, not the widest: a handful of reads queue behind other PCIe traffic and run tens of microseconds
    // long, which says nothing about how well a typical read locates the device clock.
    std::ranges::sort(bracket_widths_ns);
    const double typical_bracket_ns = bracket_widths_ns[bracket_widths_ns.size() / 2];
    const auto quantile = [&](double q) {
        return magnitudes[static_cast<size_t>(q * static_cast<double>(magnitudes.size() - 1))];
    };

    std::cout << "[ SYNC ] independent clock read (" << (mapped_stamp != nullptr ? "mapped load" : "generic read")
              << "): " << residuals_ns.size() << " reads over " << rounds_checked << " rounds, host bracket +/-"
              << typical_bracket_ns << "ns; published mapping off by " << mean_residual_ns
              << "ns on average, |residual| p50=" << quantile(0.50) << "ns p90=" << quantile(0.90)
              << "ns max=" << magnitudes.back() << "ns; RT profiler claims " << claimed_error_ns << "ns" << std::endl;

    // Averaging drives the bracket's noise down by the square root of the sample count, so the mean resolves the
    // disagreement to tens of nanoseconds; a regression in it shows up in the line printed above long before it
    // reaches the bound. The mean is not expected to be zero -- a read's true sampling instant is not the midpoint of
    // its bracket, since the two legs of a PCIe access are unequal, and that asymmetry is a constant of the
    // measurement rather than an error in the mapping.
    EXPECT_LE(std::abs(mean_residual_ns), kMaxMeanMappingErrorNs)
        << "an independent read of the device clock puts the published mapping further out than a few microseconds";
    EXPECT_LE(quantile(0.90), kMaxTailMappingErrorNs)
        << "individual reads disagree with the published mapping by more than a few microseconds";
}

}  // namespace
}  // namespace tt::tt_metal
