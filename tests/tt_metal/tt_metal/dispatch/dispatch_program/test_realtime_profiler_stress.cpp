// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stress test for the real-time (RT) profiler. If NCRISC or the host receiver can't drain as fast as
// BRISC produces, the device ring fills and BRISC drops records.
//
// The drain path is expected to absorb peak dispatch without backing up. This test replays a 4096 blank-kernel trace
// back-to-back to feed BRISC at the peak rate dispatch_s can sustain, then asserts every record arrived, the device
// ring never filled, the host D2H FIFO never filled, and there was no timestamp corruption.

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <thread>
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

#include "tt_metal/common/env_lib.hpp"
#include "tt_metal/distributed/realtime_profiler_manager.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

// Matches RT_PROFILER_RING_CAPACITY in realtime_profiler_ring_buffer.hpp.
// Picking the trace length equal to the ring capacity is the worst case for
// the back-pressure path: BRISC can fill the ring in roughly the time it
// takes NCRISC to push 1–2 entries over PCIe, so by enqueue ~80 of 4096
// the ring is at capacity and stays there for the rest of the trace.
constexpr uint32_t kNumProgramsInTrace = 4096;

constexpr uint32_t kDefaultStressReplaySeconds = 60;

constexpr uint32_t kDefaultDropAccountingSeconds = 60;

// Trace stores one EnqueueProgram dispatch packet per program. Blank-kernel
// programs with no CBs / no runtime args are tiny (~hundreds of bytes), so
// 64 MB is comfortably more than 4096 of them need; sized generously so a
// future change to the dispatch packet layout can't silently OOM the trace
// region and turn this into a flake. Lives in DRAM, not L1, so it doesn't
// interact with the worker_l1_size eligibility check we just added.
constexpr size_t kTraceRegionSize = 64 * 1024 * 1024;

// Programs in the trace use this runtime_id so every record we receive can
// be attributed to this test (records with runtime_id == 0 are reserved for
// infrastructure traffic and dropped host-side).
constexpr uint32_t kStressRuntimeId = 0xBEEFu;

// 1s upper bound per record: even a blank kernel still has dispatch_s'
// kernel_start/end pulse spread by at least a handful of cycles, but it
// should never be in the millisecond range. Anything beyond 1s means a
// timestamp got corrupted (e.g. wraparound, swapped halves) under load.
// Named distinctly from kMaxDurationNs in test_realtime_profiler_sanity.cpp
// because both files share a Unity build TU and identically-named constants
// in anonymous namespaces would collide at compile time.
constexpr double kMaxStressDurationNs = 1'000'000'000.0;

// Quiesce + drain window before unregistering the callback.
constexpr auto kPostQuiesceDrain = std::chrono::milliseconds(2000);

// Allowed slack for the deterministic startup race where the compute kernel
// detects dispatch_d's stream-register clearing before dispatch_s has
// recorded the first start_timestamp, producing one record where
// end_timestamp < start_timestamp by a handful of cycles. Same value the
// host/device correlation test tolerates (see test_realtime_profiler.py
// :: test_host_device_correlation, "startup_race_threshold") and the same
// value the production Tracy handler uses to distinguish "benign" from
// "noisy" skips (see realtime_profiler_tracy_consumer.cpp,
// kStartupRaceThreshold).
constexpr uint64_t kStartupRaceSlackCycles = 100'000;

// Hard upper bound on the fraction of records that can come back with
// end_timestamp < start_timestamp before this test fails. The production
// host receiver already silently skips any such record on the Tracy path
// (see realtime_profiler_tracy_consumer.cpp:HandleRecord), so a tiny number
// of these is tolerated by the live system; we only want this test to flag
// a *regression* where the corruption rate balloons (e.g. an off-by-one in
// rt_ring_full leading to systemic slot reuse). Empirically the live
// system produces ~1 such record per ~4000 launches on Blackhole p100a
// (rate ≈ 0.025%); 1% gives ~40x headroom over the observed baseline.
constexpr double kMaxBadTimestampFraction = 0.01;

distributed::MeshWorkload build_blank_kernel_workload(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    Program program = CreateProgram();

    // Blank kernels on BRISC + NCRISC + TRISC on a single core. Single-core
    // minimizes dispatch payload (one launch_msg per RISC, one core's worth
    // of kernel-config state) so the dispatch_s -> RT-profiler mailbox
    // pulse rate is dominated by trace cmd consumption, not program
    // launch overhead.
    const CoreCoord stress_core{0, 0};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        stress_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        stress_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernel(program, "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp", stress_core, ComputeConfig{});

    program.set_runtime_id(static_cast<uint64_t>(kStressRuntimeId));

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    return workload;
}

std::shared_ptr<distributed::MeshDevice> open_full_mesh() {
    return distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        kTraceRegionSize,
        1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
}

std::shared_ptr<distributed::MeshDevice> open_unit_mesh() {
    return distributed::MeshDevice::create_unit_mesh(
        0, DEFAULT_L1_SMALL_SIZE, kTraceRegionSize, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
}

TEST(RealtimeProfilerStress, PeakLoadPreservesRecords) {
    auto mesh_device = open_full_mesh();
    ASSERT_NE(mesh_device, nullptr);

    // RT profiler activation is decided during the init-sync handshake at
    // mesh open, so by the time the mesh is opened this query is
    // stable. When false, the dispatch config (ETH dispatch, non-MMIO
    // chip, kernels nullified, no valid RT core, worker_l1_size shrunk
    // below the ring size, ...) leaves RT profiler off; the test has
    // nothing to assert in that case so it skips cleanly.
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }
    const auto* rt = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt, nullptr);
    const uint64_t num_active_devices = rt->num_active_devices();

    uint64_t stress_records = 0;
    uint64_t startup_race_skips = 0;
    uint64_t large_negative_skips = 0;
    uint64_t bad_frequency = 0;
    uint64_t implausible_duration = 0;
    int64_t worst_negative_delta = 0;
    uint64_t max_callback_batch = 0;
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
            max_callback_batch = std::max<uint64_t>(max_callback_batch, batch.records.size());
            for (const auto& rec : batch.records) {
                if (rec.runtime_id != kStressRuntimeId) {
                    continue;
                }
                ++stress_records;
                if (rec.end_timestamp < rec.start_timestamp) {
                    const uint64_t neg_delta = rec.start_timestamp - rec.end_timestamp;
                    if (neg_delta <= kStartupRaceSlackCycles) {
                        ++startup_race_skips;
                    } else {
                        ++large_negative_skips;
                        worst_negative_delta = std::min(-static_cast<int64_t>(neg_delta), worst_negative_delta);
                    }
                }
                if (!(rec.frequency > 0.0)) {
                    ++bad_frequency;
                } else if (rec.end_timestamp >= rec.start_timestamp) {
                    const double duration_ns =
                        static_cast<double>(rec.end_timestamp - rec.start_timestamp) / rec.frequency;
                    if (duration_ns >= kMaxStressDurationNs) {
                        ++implausible_duration;
                    }
                }
            }
        });

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);

    // Compile + warm up the workload outside the trace capture so the trace
    // contains only steady-state dispatch packets (no compile/upload hops).
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    // Capture: 4096 back-to-back EnqueueMeshWorkload calls of the same
    // blank-kernel workload. Reusing one workload (vs. building 4096
    // distinct programs) keeps compile time near zero and avoids host-side
    // memory pressure; the dispatch commands captured in the trace are
    // independent per-enqueue, so dispatch_s still fires 4096 separate
    // kernel_start pulses on replay.
    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < kNumProgramsInTrace; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    const std::chrono::seconds replay_window(
        tt::parse_env<std::uint32_t>("TT_RT_PROFILER_SATURATION_SECONDS", kDefaultStressReplaySeconds));
    uint64_t num_replays = 0;
    const auto replay_deadline = std::chrono::steady_clock::now() + replay_window;
    do {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
        ++num_replays;
    } while (std::chrono::steady_clock::now() < replay_deadline);

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    const uint32_t peak_fifo_pages = rt->peak_fifo_pages();
    const uint32_t fifo_capacity_pages = rt->host_fifo_capacity_pages();
    const uint32_t ring_full_waits = rt->ring_full_wait_count();
    const uint64_t published_batches = rt->num_published_batches();
    const double mean_publish_batch =
        published_batches ? static_cast<double>(rt->num_published_records()) / published_batches : 0.0;
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(trace_id);

    const uint64_t expected_stress_records =
        static_cast<uint64_t>(kNumProgramsInTrace) * num_replays * num_active_devices;

    log_info(
        tt::LogTest,
        "[RT profiler stress] {} stress records across {} active device(s) over {} replays, max_callback_batch={}, "
        "mean_publish_batch={:.1f}, peak_fifo={}/{} pages, ring_full_waits={}, {} startup-race skips, {} "
        "large-negative-delta skips (worst delta = {} cycles), {} bad-frequency, {} implausible-duration",
        stress_records,
        num_active_devices,
        num_replays,
        max_callback_batch,
        mean_publish_batch,
        peak_fifo_pages,
        fifo_capacity_pages,
        ring_full_waits,
        startup_race_skips,
        large_negative_skips,
        worst_negative_delta,
        bad_frequency,
        implausible_duration);

    ASSERT_GE(stress_records, expected_stress_records)
        << "expected one record per program run: " << kNumProgramsInTrace << " programs per replay x " << num_replays
        << " replays x " << num_active_devices
        << " active device(s). A shortfall means profiler records were dropped at some point in the pipeline.";

    EXPECT_LT(peak_fifo_pages, fifo_capacity_pages)
        << "host D2H FIFO reached capacity; the receiver drained it slower than the device filled it";

    EXPECT_EQ(ring_full_waits, 0u)
        << "device ring reached capacity; the receiver drained it slower than the device filled it";

    const uint64_t max_allowed_large_negative =
        static_cast<uint64_t>(static_cast<double>(stress_records) * kMaxBadTimestampFraction);
    EXPECT_LE(large_negative_skips, max_allowed_large_negative)
        << large_negative_skips << " stress record(s) had end_timestamp < start_timestamp by more than "
        << kStartupRaceSlackCycles << " cycles, exceeding the allowed budget of " << max_allowed_large_negative
        << " (= " << (kMaxBadTimestampFraction * 100.0) << "% of " << stress_records << " stress records). "
        << "These are torn 64-bit reads or stale-slot residue from BRISC writing the timestamp slot before "
        << "bumping write_index; the production Tracy handler silently drops them. A spike here means the "
        << "corruption rate has become systemic — most likely an off-by-one in the rt_ring_full check or a "
        << "missing memory barrier between slot write and write_index increment.";
    EXPECT_EQ(bad_frequency, 0u) << bad_frequency << " stress record(s) had a non-positive frequency";
    EXPECT_EQ(implausible_duration, 0u) << implausible_duration
                                        << " stress record(s) reported duration >= " << kMaxStressDurationNs
                                        << " ns (clock corruption / mis-decoded timestamp)";

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerStress, CallbackDeliveryLatency) {
    using namespace std::chrono_literals;
    constexpr uint32_t kPacedId = 0x6AC0;
    constexpr std::array<std::chrono::microseconds, 5> kGaps = {5us, 50us, 200us, 1000us, 5000us};
    constexpr uint32_t kOpsPerGap = 100;
    constexpr double kMaxPacedOverheadP50Us = 500.0;
    constexpr double kMaxPacedOverheadP99Us = 20'000.0;

    constexpr uint32_t num_gaps = static_cast<uint32_t>(kGaps.size());
    constexpr uint32_t total_paced = kOpsPerGap * num_gaps;

    auto mesh_device = open_unit_mesh();
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<std::chrono::steady_clock::time_point> paced_enqueued(total_paced);
    std::vector<std::atomic<std::chrono::steady_clock::rep>> paced_delivered(total_paced);
    std::atomic<uint64_t> paced_idx{0};
    std::atomic<uint64_t> dropped_total{0};

    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
            dropped_total.fetch_add(batch.dropped, std::memory_order_relaxed);
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            for (const auto& rec : batch.records) {
                if (rec.runtime_id == kPacedId) {
                    const uint64_t idx = paced_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx < total_paced) {
                        paced_delivered[idx].store(now, std::memory_order_relaxed);
                    }
                }
            }
        });

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(cq, workload, true);

    for (auto& [_, prog] : workload.get_programs()) {
        prog.set_runtime_id(static_cast<uint64_t>(kPacedId));
    }
    distributed::MeshTraceId paced_trace = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(cq.id(), paced_trace);

    uint32_t k = 0;
    for (uint32_t gap_idx = 0; gap_idx < num_gaps; ++gap_idx) {
        mesh_device->quiesce_devices();
        const auto gap = kGaps[gap_idx];
        const auto bucket_start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < kOpsPerGap; ++i) {
            while (std::chrono::steady_clock::now() < bucket_start + gap * i) {
            }
            mesh_device->replay_mesh_trace(cq.id(), paced_trace, false);
            paced_enqueued[k] = std::chrono::steady_clock::now();
            ++k;
        }
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(paced_trace);

    auto percentile = [](std::vector<double>& v, double p) {
        if (v.empty()) {
            return 0.0;
        }
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, static_cast<size_t>(std::lround(p * static_cast<double>(v.size() - 1))))];
    };

    const uint64_t paced_matched = std::min<uint64_t>(paced_idx.load(), total_paced);
    double worst_paced_overhead_p50_us = 0.0;
    double worst_paced_overhead_p99_us = 0.0;
    for (uint32_t gap_idx = 0; gap_idx < num_gaps; ++gap_idx) {
        std::vector<double> overhead_us;
        overhead_us.reserve(kOpsPerGap);
        for (uint32_t i = 0; i < kOpsPerGap; ++i) {
            const uint32_t idx = gap_idx * kOpsPerGap + i;
            const auto d = paced_delivered[idx].load(std::memory_order_relaxed);
            if (d == 0) {
                continue;
            }
            const auto delivered_tp = std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(d));
            const double lat = std::chrono::duration<double, std::micro>(delivered_tp - paced_enqueued[idx]).count();
            overhead_us.push_back(std::max(0.0, lat - static_cast<double>(kGaps[gap_idx].count())));
        }
        const double ov_p50 = percentile(overhead_us, 0.50);
        const double ov_p99 = percentile(overhead_us, 0.99);
        worst_paced_overhead_p50_us = std::max(worst_paced_overhead_p50_us, ov_p50);
        worst_paced_overhead_p99_us = std::max(worst_paced_overhead_p99_us, ov_p99);
        log_info(
            tt::LogTest,
            "[RT profiler stress] gap={:5}us | overhead p50={:.1f} p99={:.1f} max={:.1f}us",
            kGaps[gap_idx].count(),
            ov_p50,
            ov_p99,
            percentile(overhead_us, 1.0));
    }

    EXPECT_EQ(dropped_total.load(), 0u)
        << "callback dropped records; deliveries are paired to enqueues by position, so a drop misaligns every "
        << "later pair and the latencies below are meaningless";
    EXPECT_GE(paced_matched, total_paced - total_paced / 100)
        << "too few of the " << total_paced << " paced ops reached the callback; the latency percentiles "
        << "below are over a partial sample and unreliable";
    EXPECT_LT(worst_paced_overhead_p50_us, kMaxPacedOverheadP50Us)
        << "median delivery overhead too high; the consumer is not waking promptly (a fixed "
        << "backoff/oversleep would show up here)";
    EXPECT_LT(worst_paced_overhead_p99_us, kMaxPacedOverheadP99Us)
        << "tail delivery overhead too high; occasional long stalls in the delivery path";
    EXPECT_TRUE(mesh_device->close());
}

// Three consumers read the same record stream at different throttled rates. Verifies the per-reader
// drop accounting: for every consumer, received + dropped covers every record produced, and a
// throttled consumer drops no more than its sustain rate forces (no over-dropping).
TEST(RealtimeProfilerStress, ConsumerDropAccountingUnderLoad) {
    const std::chrono::seconds run_window(
        tt::parse_env<std::uint32_t>("TT_RT_PROFILER_DROP_ACCOUNTING", kDefaultDropAccountingSeconds));

    auto mesh_device = open_full_mesh();
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }
    const uint64_t num_devices = mesh_device->num_devices();
    const auto* rt = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt, nullptr);

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(cq, workload, true);

    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < kNumProgramsInTrace; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    constexpr auto kCalibrationWindow = std::chrono::seconds(2);
    const uint64_t pubs_before = rt->num_published_records();
    const auto cal_start = std::chrono::steady_clock::now();
    const auto cal_deadline = cal_start + kCalibrationWindow;
    while (std::chrono::steady_clock::now() < cal_deadline) {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
    }
    const double cal_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - cal_start).count();
    const double production_rate = static_cast<double>(rt->num_published_records() - pubs_before) / cal_seconds;
    ASSERT_GT(production_rate, 0.0) << "no records produced during calibration";
    mesh_device->quiesce_devices();

    // fraction of the production rate each consumer can sustain
    constexpr double kBorderlineSustainFraction = 0.95;
    constexpr double kSlowSustainFraction = 0.1;

    const auto borderline_per_record =
        std::chrono::nanoseconds(static_cast<int64_t>(1e9 / (production_rate * kBorderlineSustainFraction)));
    const auto slow_per_record =
        std::chrono::nanoseconds(static_cast<int64_t>(1e9 / (production_rate * kSlowSustainFraction)));
    log_info(
        tt::LogTest,
        "[RT profiler stress] measured production {:.0f} rec/s across {} device(s); borderline={}ns/rec slow={}ns/rec",
        production_rate,
        num_devices,
        borderline_per_record.count(),
        slow_per_record.count());

    struct Counters {
        std::atomic<uint64_t> received{0};
        std::atomic<uint64_t> dropped{0};
    };
    Counters keeps_up;
    Counters borderline;
    Counters slow;

    auto make_consumer = [](Counters& c, std::chrono::nanoseconds per_record) {
        return [&c, per_record, start = std::chrono::steady_clock::time_point{}, paced = uint64_t{0}](
                   const ProgramRealtimeRecordBatch& batch) mutable {
            c.received.fetch_add(batch.records.size(), std::memory_order_relaxed);
            c.dropped.fetch_add(batch.dropped, std::memory_order_relaxed);
            if (per_record == std::chrono::nanoseconds::zero()) {
                return;
            }
            if (paced == 0) {
                start = std::chrono::steady_clock::now();
            }
            paced += batch.records.size();
            const auto deadline = start + per_record * paced;
            while (std::chrono::steady_clock::now() < deadline) {
            }
        };
    };

    ProgramRealtimeProfilerCallbackHandle h_keeps_up =
        RegisterProgramRealtimeProfilerCallback(make_consumer(keeps_up, std::chrono::nanoseconds::zero()));
    ProgramRealtimeProfilerCallbackHandle h_borderline =
        RegisterProgramRealtimeProfilerCallback(make_consumer(borderline, borderline_per_record));
    ProgramRealtimeProfilerCallbackHandle h_slow =
        RegisterProgramRealtimeProfilerCallback(make_consumer(slow, slow_per_record));

    const auto run_deadline = std::chrono::steady_clock::now() + run_window;
    while (std::chrono::steady_clock::now() < run_deadline) {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);

    const uint32_t peak_fifo_pages = rt->peak_fifo_pages();
    const uint32_t fifo_capacity_pages = rt->host_fifo_capacity_pages();
    const uint64_t published_batches = rt->num_published_batches();
    const double mean_publish_batch =
        published_batches ? static_cast<double>(rt->num_published_records()) / published_batches : 0.0;

    const uint64_t keeps_up_received = keeps_up.received.load();
    const uint64_t keeps_up_dropped = keeps_up.dropped.load();
    ASSERT_GT(keeps_up_received, 0u) << "no records delivered; cannot assess accounting";
    ASSERT_EQ(keeps_up_dropped, 0u)
        << "unthrottled consumer dropped; it does no per-record work, so this likely means host contention "
        << "starved its callback thread";

    auto accounted = [](const Counters& c) { return c.received.load() + c.dropped.load(); };
    auto wait_until_accounted = [&accounted](const Counters& c, uint64_t target) {
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (accounted(c) < target && std::chrono::steady_clock::now() < give_up) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    };
    wait_until_accounted(borderline, keeps_up_received);
    wait_until_accounted(slow, keeps_up_received);

    const uint64_t borderline_received = borderline.received.load();
    const uint64_t borderline_dropped = borderline.dropped.load();
    const uint64_t slow_received = slow.received.load();
    const uint64_t slow_dropped = slow.dropped.load();
    const uint64_t borderline_total = borderline_received + borderline_dropped;
    const uint64_t slow_total = slow_received + slow_dropped;

    log_info(
        tt::LogTest,
        "[RT profiler stress] devices={} total={} peak_fifo={}/{} pages mean_publish_batch={:.1f} | "
        "borderline: recv={} drop={} sum={} | slow: recv={} drop={} sum={}",
        num_devices,
        keeps_up_received,
        peak_fifo_pages,
        fifo_capacity_pages,
        mean_publish_batch,
        borderline_received,
        borderline_dropped,
        borderline_total,
        slow_received,
        slow_dropped,
        slow_total);

    UnregisterProgramRealtimeProfilerCallback(h_keeps_up);
    UnregisterProgramRealtimeProfilerCallback(h_borderline);
    UnregisterProgramRealtimeProfilerCallback(h_slow);
    mesh_device->release_mesh_trace(trace_id);

    EXPECT_LT(peak_fifo_pages, fifo_capacity_pages)
        << "host D2H FIFO reached capacity; the receiver drained it slower than the device filled it";

    EXPECT_EQ(borderline_total, keeps_up_received) << "borderline consumer lost or double-counted records";
    EXPECT_EQ(slow_total, keeps_up_received) << "slow consumer lost or double-counted records";
    EXPECT_LE(borderline_dropped, slow_dropped)
        << "the faster (borderline) consumer dropped more than the slower one; impossible unless the ring is "
        << "misattributing drops between the two readers";

    constexpr double kMaxDeliveryShortfall = 0.2;
    constexpr double kMinOverdropTolerance = 0.03;
    auto expect_no_overdrop =
        [kMinOverdropTolerance](const char* name, uint64_t dropped, uint64_t total, double sustain_fraction) {
            const double drop_frac = static_cast<double>(dropped) / static_cast<double>(total);
            const double max_drop =
                (1.0 - sustain_fraction) + std::max(kMinOverdropTolerance, sustain_fraction * kMaxDeliveryShortfall);
            log_info(
                tt::LogTest,
                "[RT profiler stress] {} dropped {:.1f}% (over-drop limit {:.1f}%, {} of {})",
                name,
                100.0 * drop_frac,
                100.0 * max_drop,
                dropped,
                total);
            EXPECT_LE(drop_frac, max_drop)
                << name << " dropped past the over-drop limit; the ring overwrote records it had capacity to take";
        };
    expect_no_overdrop("borderline", borderline_dropped, borderline_total, kBorderlineSustainFraction);
    expect_no_overdrop("slow", slow_dropped, slow_total, kSlowSustainFraction);

    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
