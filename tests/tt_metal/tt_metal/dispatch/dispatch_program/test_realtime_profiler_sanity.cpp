// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merge-gate sanity coverage for the real-time (RT) profiler, in three layers:
//
//   RealtimeProfilerSanity -- the record service and its consumer threads, driven through hand-fed rings. A MeshDevice
//       contributes exactly one ring, so multi-ring and mid-callback behaviour is unreachable from a device test.
//   RealtimeProfilerDeviceSanity -- the whole pipeline on a unit mesh: mailbox layout, D2H socket init, clock
//       bring-up, kernel source propagation, timestamp extraction.
//
// Runs as part of `tt-metalium-validation-basic` (merge-gate `metalium-basic-tests` job) on both N150 (WH) and
// P150b (BH). The device layer skips via IsProgramRealtimeProfilerActive() where the profiler can't be enabled.

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

#include <umd/device/chip_helpers/tlb_manager.hpp>
#include <umd/device/pcie/tlb_handle.hpp>
#include <umd/device/pcie/tlb_window.hpp>
#include <umd/device/types/xy_pair.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "impl/device/device_manager.hpp"
#include "impl/realtime_profiler/realtime_profiler_clock_sync.hpp"
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
// Sanity cap only, to catch a broken clock / mis-decoded timestamp: the 40K-NOP kernels below stay in the
// tens-of-microseconds range even on slow silicon.
constexpr double kMaxDurationNs = 1'000'000'000.0;

// Asserted as a distribution rather than a per-record ceiling: a degraded read path lifts the median, while a
// re-anchor policy that stops rejecting wide brackets only fattens the tail.
//
// Sized for the slower of the two architectures -- Blackhole reads the counter through its own TLB window and reports
// ~0.4us, Wormhole falls back to UMD's generic register read where it cannot.
constexpr uint64_t kSyncErrorP50Ns = 6'000;
constexpr uint64_t kSyncErrorP90Ns = 10'000;
constexpr uint64_t kSyncErrorP99Ns = 15'000;

constexpr const char* kSourceMarkerPrefix = "rt_profiler_marker_";

// Inlined (200x200 unrolled NOPs) rather than loaded from a file under tt_metal/programming_examples/...: those files
// ship in the `metalium-examples` deb, while this test runs from `tt-metalium-validation` in CI. The NOP count is
// load-bearing for the implausible-duration check: a corrupted timestamp (e.g. swapped 32-bit halves) would still
// satisfy end > start for a ns-scale kernel, but surfaces as a multi-second duration here.
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

// runtime_id == 0 is reserved for infrastructure traffic and filtered host-side.
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

// Linearly interpolated between adjacent ranks; nearest-rank would snap each percentile onto one sample instead.
// `sorted` must be ascending and non-empty. Prefixed: this file shares a Unity build TU with the rest of the
// dispatch tests.
double rt_percentile(const std::vector<double>& sorted, double p) {
    const double rank = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<size_t>(rank);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (rank - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
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

// Opens a unit mesh with the RT profiler active, or skips. The profiler attaches its record ring during mesh open, so
// the check is stable by the time create_unit_mesh returns.
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

// Synthetic steady_clock instants. Real ones are ~1e15 ns since boot, close enough to the 2^53 limit of exact integer
// arithmetic in a double that the model's own rounding would show up in the assertions below.
constexpr std::chrono::steady_clock::time_point host_instant(int64_t ns) {
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(ns));
}

// A probe at `at` reading a clock that has been running at `rate` ticks/ns since tick zero of the session.
RealtimeProfilerClockSync::Anchor anchor_at(
    std::chrono::steady_clock::time_point at, double rate, std::chrono::nanoseconds bracket) {
    return RealtimeProfilerClockSync::Anchor{
        .host = at,
        .ticks = static_cast<uint64_t>(rate * static_cast<double>(at.time_since_epoch().count())),
        .bracket = bracket};
}

constexpr auto kProbeBracket = std::chrono::nanoseconds(700);
constexpr double kNominalRate = 1.35;

// A real chord spans one sync interval, and the quantities these tests check -- the slope's own noise, what counts as
// an implausible rate, what a step is worth as curvature -- are all fractions of that span. Deriving them keeps the
// tests meaningful at whatever interval is configured rather than only at the one they were written against.
double chord_span_ns() { return static_cast<double>(RealtimeProfilerClockSync::sync_interval().count()); }
double chord_rate_noise() { return 2.0 * static_cast<double>(kProbeBracket.count()) / chord_span_ns(); }
RealtimeProfilerClockSync::Anchor chord_close(const RealtimeProfilerClockSync::Anchor& open) {
    return anchor_at(open.host + RealtimeProfilerClockSync::sync_interval(), kNominalRate, kProbeBracket);
}

TEST(RealtimeProfilerChordMapping, SecantRecoversTheRateAndPlacesTheAnchorOnTheClosingProbe) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto closing = chord_close(open);

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, std::nullopt, {});

    ASSERT_TRUE(planned.has_value());
    EXPECT_NEAR(planned->frequency, kNominalRate, 1e-6);
    // The mapping must reproduce the closing probe exactly: that probe is the one thing here that was measured.
    const double closing_host_ns =
        (static_cast<double>(closing.ticks) - static_cast<double>(planned->mapping.device_cycle_offset)) /
        planned->frequency;
    EXPECT_NEAR(closing_host_ns, static_cast<double>(closing.host.time_since_epoch().count()), 1.0);
}

// A slope is only as good as the span it was measured over; below half an interval the previous one is better.
TEST(RealtimeProfilerChordMapping, ChordTooShortToTakeASlopeFromIsRefused) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto barely_later =
        anchor_at(open.host + RealtimeProfilerClockSync::sync_interval() / 4, kNominalRate, kProbeBracket);

    EXPECT_FALSE(RealtimeProfilerClockSync::plan_chord_mapping(open, barely_later, std::nullopt, {}).has_value());
}

// The counter is read low word first and that latches the high word, so a composed timestamp cannot tear and a pair is
// only checked for the two things that make a secant undefined. Ticks that did not advance is one of them.
TEST(RealtimeProfilerChordMapping, NonMonotonicTicksAreRefused) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    auto closing = chord_close(open);
    closing.ticks = open.ticks;

    EXPECT_FALSE(RealtimeProfilerClockSync::plan_chord_mapping(open, closing, std::nullopt, {}).has_value());
}

// A read serviced late puts its anchor at the midpoint of a wide bracket, which moves the span the slope is taken over
// without touching the ticks -- so the slope it produces looks wrong by (bracket/2)/span while the clock did nothing at
// all. Refusing that pair is refusing a measurement for being imprecise, and nothing retries a refused pair, so it
// stalls the device behind its own oldest record. It is placed, and charged for its worst anchor instead.
TEST(RealtimeProfilerChordMapping, LateServicedReadIsPlacedAndChargedForItsBracket) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto interval = RealtimeProfilerClockSync::sync_interval();
    // 30% of the span, which puts the apparent slope ~26% off a clock that never moved. The ticks are exactly what this
    // clock produces in an interval: the only thing wrong with this probe is when it was serviced.
    const auto late_bracket = std::chrono::nanoseconds(static_cast<int64_t>(0.3 * chord_span_ns()));
    const RealtimeProfilerClockSync::Anchor serviced_late{
        .host = open.host + interval + late_bracket / 2,
        .ticks = open.ticks + static_cast<uint64_t>(kNominalRate * chord_span_ns()),
        .bracket = late_bracket};

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, serviced_late, std::nullopt, {});

    ASSERT_TRUE(planned.has_value());
    EXPECT_GE(planned->mapping.sync_error, late_bracket / 2);
}

// A measured chord may be reused only up to its far anchor: past that the pair that brackets a timestamp is a
// different, tighter one. The publish loop batches on this, and it has to admit the timestamp the chord was resolved
// for or the loop makes no progress.
TEST(RealtimeProfilerChordMapping, AMeasuredChordIsReusableUpToItsFarAnchor) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto closing = chord_close(open);

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, std::nullopt, {});

    ASSERT_TRUE(planned.has_value());
    EXPECT_EQ(planned->batch_through_ticks, closing.ticks);
    EXPECT_EQ(planned->close_ticks, closing.ticks);
}

// A timestamp inside the chord rides a secant pinned at both ends, so the endpoints bound it. Outside, the slope is
// extrapolated and its uncertainty compounds with distance; a chord's own bound says nothing about that.
TEST(RealtimeProfilerChordMapping, ExtrapolatingPastTheChordIsChargedForTheDistance) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto closing = chord_close(open);
    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, std::nullopt, {});
    ASSERT_TRUE(planned.has_value());

    const uint64_t inside = open.ticks + (closing.ticks - open.ticks) / 2;
    EXPECT_EQ(
        RealtimeProfilerClockSync::place_on_chord(*planned, inside).sync_error,
        RealtimeProfilerClockSync::interpolation_error(open, closing));

    // One second before the chord opens. At the chord's own slope noise that is milliseconds of uncertainty, and the
    // endpoint term alone would have claimed sub-microsecond.
    const uint64_t far_before = open.ticks - static_cast<uint64_t>(kNominalRate * 1e9);
    const auto far_error = RealtimeProfilerClockSync::place_on_chord(*planned, far_before).sync_error;
    EXPECT_GT(far_error, std::chrono::microseconds(100));
    EXPECT_NEAR(static_cast<double>(far_error.count()), 1e9 * chord_rate_noise(), 1e9 * chord_rate_noise() * 0.05);
}

// The published error must cover the two placements the chord is pinned by, whatever the clock did between them.
TEST(RealtimeProfilerChordMapping, ErrorCoversBothEndpointPlacements) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, std::chrono::nanoseconds(400));
    const auto closing =
        anchor_at(open.host + RealtimeProfilerClockSync::sync_interval(), kNominalRate, std::chrono::nanoseconds(900));

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, std::nullopt, {});

    ASSERT_TRUE(planned.has_value());
    EXPECT_GE(planned->mapping.sync_error, std::chrono::nanoseconds(450));
}

// A probe inside a chord was not fitted to it, so its distance from the secant is the clock's own departure. This is
// the term that used to be inferred from how much two chords' slopes differed -- which read zero whenever a chord was
// resolved twice, since it was then compared against itself.
TEST(RealtimeProfilerChordMapping, ClockDepartureMeasuredAtAnInteriorProbeIsCharged) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto closing = chord_close(open);
    // A probe at the chord's midpoint, sitting a clear 3us off the secant: far past what its own 700ns bracket
    // explains.
    constexpr auto kDeparture = std::chrono::microseconds(3);
    auto interior = anchor_at(open.host + RealtimeProfilerClockSync::sync_interval() / 2, kNominalRate, kProbeBracket);
    interior.host += kDeparture;

    const auto bow = RealtimeProfilerClockSync::departure_from_chord(open, closing, interior);
    EXPECT_NEAR(
        static_cast<double>(bow.count()),
        static_cast<double>((kDeparture - kProbeBracket / 2).count()),
        static_cast<double>(kProbeBracket.count()));

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, std::nullopt, bow);
    ASSERT_TRUE(planned.has_value());
    EXPECT_EQ(planned->mapping.sync_error, RealtimeProfilerClockSync::interpolation_error(open, closing) + bow);
}

// A probe landing off the secant by less than its own read could have misplaced it says nothing about the clock.
// Charging it would put invented error on every record.
TEST(RealtimeProfilerChordMapping, DepartureWithinAProbesOwnReadNoiseIsNotCharged) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto closing = chord_close(open);
    auto interior = anchor_at(open.host + RealtimeProfilerClockSync::sync_interval() / 2, kNominalRate, kProbeBracket);
    interior.host += kProbeBracket / 4;  // inside its own half-bracket

    EXPECT_EQ(
        RealtimeProfilerClockSync::departure_from_chord(open, closing, interior), std::chrono::nanoseconds::zero());
}

// A chord with nothing inside it has no evidence either way, and absence of evidence is not a bow.
TEST(RealtimeProfilerChordMapping, AChordWithNoInteriorProbeReportsOnlyItsEndpoints) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto closing = chord_close(open);

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, std::nullopt, {});

    ASSERT_TRUE(planned.has_value());
    EXPECT_EQ(planned->mapping.sync_error, RealtimeProfilerClockSync::interpolation_error(open, closing));
}

// The rate a record is published with is measured across the baseline, not across its own chord: a 100us chord's slope
// carries thousands of ppm, and every duration a consumer computes divides by it.
TEST(RealtimeProfilerChordMapping, PublishedRateComesFromTheBaselineNotTheChord) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    auto closing = chord_close(open);
    // Well inside what two 700ns brackets over this span can produce.
    closing.ticks += static_cast<uint64_t>(0.3 * chord_rate_noise() * kNominalRate * chord_span_ns());
    const RealtimeProfilerClockSync::BaselineRate baseline{.rate = kNominalRate, .noise = 0.0005};

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, baseline, {});

    ASSERT_TRUE(planned.has_value());
    EXPECT_NEAR(planned->frequency, kNominalRate, 1e-9);
    // The chord's own slope is still reported: it is the local rate, so it is what the next interval's curvature term
    // has to compare against.
    EXPECT_GT(planned->chord_rate, kNominalRate);
}

// Publishing a baseline-wide rate must not move where a record lands: each record is anchored to its own placement on
// the chord, so a rate measured somewhere else costs the record's start nothing at all.
TEST(RealtimeProfilerChordMapping, RecordLandsOnTheChordWhateverRateIsPublished) {
    const auto open = anchor_at(host_instant(1'000'000'000), kNominalRate, kProbeBracket);
    const auto closing = chord_close(open);
    // A baseline 1% away from this chord: far more than any real DVFS step, so if placement were carried at the
    // published rate this would land a record ~500ns off.
    const RealtimeProfilerClockSync::BaselineRate baseline{.rate = kNominalRate * 1.01, .noise = 0.0005};

    const auto planned = RealtimeProfilerClockSync::plan_chord_mapping(open, closing, baseline, {});

    ASSERT_TRUE(planned.has_value());
    // A record three quarters of the way through the chord.
    const uint64_t start_ticks = open.ticks + 3 * (closing.ticks - open.ticks) / 4;
    const int64_t offset = RealtimeProfilerClockSync::device_cycle_offset_for(*planned, start_ticks);
    const double mapped_host_ns = (static_cast<double>(start_ticks) - static_cast<double>(offset)) / planned->frequency;
    const double chord_host_ns = static_cast<double>(open.host.time_since_epoch().count()) +
                                 static_cast<double>(start_ticks - open.ticks) / planned->chord_rate;
    EXPECT_NEAR(mapped_host_ns, chord_host_ns, 1.0);
    // And nothing beyond the two placements is charged for it.
    EXPECT_EQ(planned->mapping.sync_error, RealtimeProfilerClockSync::interpolation_error(open, closing));
}

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
            << "RT record sync_error should be set by the init sync handshake (runtime_id=" << rec.runtime_id << ")";

        EXPECT_LT(rec.duration().count(), kMaxDurationNs)
            << "RT record duration is implausibly large (runtime_id=" << rec.runtime_id << ", chip=" << rec.chip_id
            << ", duration_ns=" << rec.duration().count() << ")";
    }

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

// Spreads programs across many servo re-anchor intervals so records sample clock_sync.sync_error over the session,
// then asserts its distribution (see kSyncErrorP50Ns et al.) stays tight.
TEST_F(RealtimeProfilerDeviceSanity, SyncAccuracy) {
    RecordCollector collector;
    // A fixed runtime_id keeps the kernel source (and its JIT compile) shared across iterations.
    constexpr uint32_t kIterations = 300;
    for (uint32_t i = 0; i < kIterations; ++i) {
        enqueue_sanity_program(mesh_device_, /*runtime_id=*/1, all_cores());
        std::this_thread::sleep_for(RealtimeProfilerClockSync::sync_interval());
    }
    quiesce_and_wait_for([&] { return collector.records().size() >= kIterations; });
    collector.stop();

    // Per record, not per anchor: anchors are few (only taken when a probe finds real drift) and their count says
    // nothing about sync quality; the error every record carries is what a consumer sees.
    std::vector<double> errors;
    const auto& records = collector.records();
    errors.reserve(records.size());
    for (const auto& record : records) {
        errors.push_back(static_cast<double>(record.clock_sync.sync_error.count()));
    }
    ASSERT_FALSE(errors.empty());
    std::ranges::sort(errors);
    const auto pct = [&errors](double p) { return rt_percentile(errors, p); };
    log_info(
        tt::LogTest,
        "[RT profiler sanity] sync_error (ns) over {} records: min={:.0f} p50={:.1f} p75={:.1f} "
        "p90={:.1f} p95={:.1f} p99={:.1f} max={:.0f}",
        errors.size(),
        errors.front(),
        pct(0.50),
        pct(0.75),
        pct(0.90),
        pct(0.95),
        pct(0.99),
        errors.back());

    EXPECT_GT(errors.front(), 0.0) << "sync_error should be populated once the clock is anchored";
    EXPECT_LT(pct(0.50), static_cast<double>(kSyncErrorP50Ns))
        << "median sync error too high; the handshake is systematically degraded";
    EXPECT_LT(pct(0.90), static_cast<double>(kSyncErrorP90Ns)) << "p90 sync error too high";
    EXPECT_LT(pct(0.99), static_cast<double>(kSyncErrorP99Ns))
        << "tail sync error too high; a bad re-anchor is not being rejected";
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

    // Capture cannot load binaries, so warm up (under kWarmupRuntimeId) before capture, then switch to
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

// Independent check of the clock mapping the RT profiler publishes on every record. Shares no code or mechanism with
// production sync: production has the host drive a round trip and bracket a device timestamp inside it, while this
// reads the device's own clock and brackets that read between two host clock reads.
//
// Nothing is fitted here -- a record's device_cycle_offset/frequency already states the mapping, so only its
// residual is measured. A single read can't resolve it (the host bracket is wider than the error sought); averaging
// many does, to about one bracket over sqrt(sample count).
//
// The kernel never waits on the host: it stamps WALL_CLOCK into one L1 slot in a bounded loop and exits, so the host
// can stop reading (or fail an assertion) at any time without stranding a spinning kernel and hanging Finish.
constexpr uint32_t kClockStampIterations = 200'000'000;  // ~5s of stamping; only reached if the stop flag never lands
constexpr uint32_t kClockCheckRounds = 3;
constexpr uint32_t kClockReadsPerRound = 500;
constexpr uint32_t kFirstClockCheckRuntimeId = 9101;
// Fixed rather than derived from the record's own sync_error -- a bound computed from the claim cannot test the
// claim. Measured disagreement is ~210ns; this leaves an order of magnitude of headroom.
constexpr double kMaxMeanMappingErrorNs = 2'000.0;
// How far a read may put the mapping out *beyond what that read itself could be wrong by*. Measured zero on
// Blackhole; sized well above that so ordinary jitter can't trip it while an uncorrected DVFS step still would.
constexpr double kMaxMappingErrorBeyondReadResolutionNs = 500.0;

// Runs on every core the stamper is not using, so the reads below are taken while the part is drawing power: on
// Blackhole, DVFS only steps AICLK under load, and that step is what this test needs to see the mapping survive.
std::string make_clock_load_kernel_source() {
    return R"(
#include <cstdint>
#include "risc_common.h"

void kernel_main() {
    const uint32_t stop_addr = get_arg_val<uint32_t>(0);
    const uint32_t max_iterations = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr const uint32_t* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stop_addr);

    for (uint32_t i = 0; i < max_iterations; i++) {
#pragma GCC unroll 128
        for (int j = 0; j < 128; j++) {
            asm("nop");
        }
        if ((i & 0xFF) == 0 && stop[0] != 0) {
            break;
        }
    }
}
)";
}

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

// Blackhole exposes a static TLB window onto the core's L1, so the stamp can be read with one load instead of going
// through UMD's generic path -- only software overhead comes off, since an MMIO read is a non-posted PCIe
// transaction either way. Null where no such window exists; the caller falls back to the generic read.
volatile uint64_t* try_map_l1_word(Cluster& cluster, ChipId chip_id, CoreCoord virtual_core, uint32_t addr) {
    try {
        const tt_xy_pair tlb_core(virtual_core.x, virtual_core.y);
        auto* tlb_manager = cluster.get_driver()->get_chip(chip_id)->get_tlb_manager();
        if (tlb_manager == nullptr || !tlb_manager->is_tlb_mapped(tlb_core, addr, sizeof(uint64_t))) {
            return nullptr;
        }
        // A lookup, never an allocation: get_tlb_window throws unless UMD already mapped this core at init, and
        // is_tlb_mapped has just confirmed the mapping covers these bytes.
        auto* window = tlb_manager->get_tlb_window(tlb_core);
        if (window == nullptr) {
            return nullptr;
        }
        const uint64_t window_offset = window->get_base_address() - window->handle_ref().get_config().local_offset;
        return reinterpret_cast<volatile uint64_t*>(window->handle_ref().get_base() + addr + window_offset);
    } catch (const std::exception&) {
        return nullptr;
    }
}

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

    volatile uint64_t* mapped_stamp = try_map_l1_word(cluster, device->id(), vcore, stamp_addr);

    const auto host_now_ns = [] { return std::chrono::steady_clock::now().time_since_epoch().count(); };
    const auto read_device_ticks = [&]() -> uint64_t {
        if (mapped_stamp != nullptr) {
            return *mapped_stamp;
        }
        uint32_t words[2] = {0, 0};
        cluster.read_core(words, sizeof(words), tt_cxy_pair(device->id(), vcore), stamp_addr);
        return (static_cast<uint64_t>(words[1]) << 32) | words[0];
    };

    // Each round brackets one device clock read, then runs one program, pairing the two so the comparison below is
    // against the mapping that was live beside that read.
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

        // Same program as the stamper, so the load starts and is torn down by the same stop flag.
        const CoreRange grid = all_cores();
        CoreRangeSet load_cores = CoreRangeSet(grid).subtract(CoreRangeSet(CoreRange(core, core)));
        const std::string load_src = make_clock_load_kernel_source();
        for (auto processor : {DataMovementProcessor::RISCV_1}) {
            auto load_kernel = CreateKernelFromString(
                stamp_program,
                load_src,
                load_cores,
                DataMovementConfig{.processor = processor, .noc = NOC::RISCV_1_default});
            for (const auto& cr : load_cores.ranges()) {
                SetRuntimeArgs(stamp_program, load_kernel, cr, {stop_addr, kClockStampIterations});
            }
        }

        distributed::MeshWorkload stamp_workload;
        stamp_workload.add_program(distributed::MeshCoordinateRange(mesh_device_->shape()), std::move(stamp_program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), stamp_workload, /*blocking=*/false);

        // Wait for the slot to go non-zero rather than assume a launch latency; a timeout fails the assertion
        // instead of hanging, since the kernel ends itself either way.
        const auto launch_deadline = std::chrono::steady_clock::now() + 10s;
        while (read_device_ticks() == 0 && std::chrono::steady_clock::now() < launch_deadline) {
            std::this_thread::sleep_for(1ms);
        }
        ASSERT_NE(read_device_ticks(), 0u) << "the stamping kernel never started";

        for (uint32_t i = 0; i < kClockReadsPerRound; ++i) {
            ClockBracket bracket;
            bracket.host_before_ns = host_now_ns();
            bracket.device_ticks = read_device_ticks();
            bracket.host_after_ns = host_now_ns();
            bracket.round = round;
            brackets.push_back(bracket);
        }

        const uint32_t stop = 1;  // written to every core: the load kernels poll their own L1 for this flag
        const CoreCoord grid_size = mesh_device_->compute_with_storage_grid_size();
        for (uint32_t y = 0; y < grid_size.y; ++y) {
            for (uint32_t x = 0; x < grid_size.x; ++x) {
                const CoreCoord v = device->virtual_core_from_logical_core(CoreCoord{x, y}, CoreType::WORKER);
                cluster.write_core_immediate(&stop, sizeof(stop), tt_cxy_pair(device->id(), v), stop_addr);
            }
        }
        distributed::Finish(mesh_device_->mesh_command_queue());

        enqueue_sanity_program(mesh_device_, kFirstClockCheckRuntimeId + round, all_cores());
        distributed::Finish(mesh_device_->mesh_command_queue());
    }

    quiesce_and_wait_for([&] {
        return collector.runtime_ids_in_range(kFirstClockCheckRuntimeId + kClockCheckRounds - 1).size() >=
               kClockCheckRounds;
    });
    collector.stop();

    std::map<uint32_t, ProgramRealtimeRecord> record_by_runtime_id;
    for (const auto& record : collector.records()) {
        if (record.frequency > 0.0) {
            record_by_runtime_id.insert({record.runtime_id, record});
        }
    }

    std::vector<double> residuals_ns;
    residuals_ns.reserve(brackets.size());
    // excess_ns is what the mapping can be blamed for: a read locates the device clock to no better than half its
    // own bracket, so a residual inside that reflects a slow read, not a wrong mapping. Without subtracting it out,
    // the two are indistinguishable -- a stretched read produces |residual| ~= 0.9x its own half-bracket regardless
    // of the mapping.
    std::vector<double> excess_ns;
    std::vector<double> bracket_widths_ns;
    std::chrono::nanoseconds claimed_error{};
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
            const double residual = mapped_host_ns - bracket.host_mid_ns();
            residuals_ns.push_back(residual);
            excess_ns.push_back(std::max(0.0, std::abs(residual) - bracket.half_width_ns()));
            bracket_widths_ns.push_back(bracket.half_width_ns());
            claimed_error = std::max(claimed_error, record.clock_sync.sync_error);
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
    const auto quantile = [&magnitudes](double q) { return rt_percentile(magnitudes, q); };
    std::ranges::sort(excess_ns);
    const auto excess = [&excess_ns](double q) { return rt_percentile(excess_ns, q); };
    // Median, not widest: a handful of reads queue behind other PCIe traffic and run tens of microseconds long,
    // which says nothing about a typical read's resolution.
    std::ranges::sort(bracket_widths_ns);
    const double typical_bracket_ns = bracket_widths_ns[bracket_widths_ns.size() / 2];

    log_info(
        tt::LogTest,
        "[RT profiler sanity] independent clock read ({}): {} reads over {} rounds, host bracket +/-{}ns; published "
        "mapping off by {}ns on average, |residual| p50={}ns p90={}ns max={}ns; RT profiler claims {}ns",
        mapped_stamp != nullptr ? "mapped load" : "generic read",
        residuals_ns.size(),
        rounds_checked,
        typical_bracket_ns,
        mean_residual_ns,
        quantile(0.50),
        quantile(0.90),
        magnitudes.back(),
        claimed_error.count());
    log_info(
        tt::LogTest,
        "[RT profiler sanity] mapping error beyond each read's own resolution: p50={}ns p90={}ns p99={}ns max={}ns "
        "({} of {} reads exceeded theirs at all)",
        excess(0.50),
        excess(0.90),
        excess(0.99),
        excess_ns.back(),
        std::ranges::count_if(excess_ns, [](double e) { return e > 0.0; }),
        excess_ns.size());

    // Mean is not expected to be zero: a read's true sampling instant is not the bracket midpoint, since the two
    // legs of a PCIe access are unequal -- a constant of the measurement, not an error in the mapping.
    EXPECT_LE(std::abs(mean_residual_ns), kMaxMeanMappingErrorNs)
        << "an independent read of the device clock puts the published mapping further out than a few microseconds";
    EXPECT_LE(excess(0.99), kMaxMappingErrorBeyondReadResolutionNs)
        << "reads disagree with the published mapping by more than they could be wrong by themselves, so the "
           "disagreement is the mapping's";
    EXPECT_LE(excess_ns.back(), 4 * kMaxMappingErrorBeyondReadResolutionNs)
        << "a single read disagrees with the published mapping far beyond its own resolution";
}

}  // namespace
}  // namespace tt::tt_metal
