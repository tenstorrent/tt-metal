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
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
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
#include "tt_metal/impl/realtime_profiler/realtime_profiler_receiver.hpp"
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

constexpr uint32_t kDefaultClockDriftSeconds = 60;

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
constexpr double kMaxStressDurationNs = 1'000'000'000.0;

// Quiesce + drain window before unregistering the callback.
constexpr auto kPostQuiesceDrain = std::chrono::milliseconds(2000);

constexpr auto kStressReportInterval = std::chrono::seconds(10);

// One sync-error sample per re-anchor per chip, counted by value rather than kept in a list: sync_error only changes
// on re-anchor (~20/s/device), so a per-record list would grow unbounded across a soak, while the value itself spans
// only a handful of nanosecond steps, so counting by value stays exact.
// Named Stress* because this file shares a Unity build TU with test_realtime_profiler_sanity.cpp.
class StressSyncErrorTally {
public:
    // Call only from the record callback: the service gives each registration a single consumer thread, so the
    // unchanged-anchor fast path below needs no lock.
    void observe(const experimental::ProgramRealtimeRecord& record) {
        const int64_t offset = record.clock_sync.device_cycle_offset;
        const auto [it, inserted] = last_offset_.try_emplace(record.chip_id, offset);
        if (!inserted) {
            if (it->second == offset) {
                return;
            }
            it->second = offset;
        }
        const std::lock_guard lock(mutex_);
        ++counts_[record.clock_sync.sync_error];
    }

    // Safe to call while the callback is still running.
    std::map<std::chrono::nanoseconds, uint64_t> counts() const {
        const std::lock_guard lock(mutex_);
        return counts_;
    }

private:
    std::unordered_map<uint32_t, int64_t> last_offset_;  // consumer thread only
    mutable std::mutex mutex_;
    std::map<std::chrono::nanoseconds, uint64_t> counts_;
};

// `sorted` must be ascending and non-empty. Named Stress* because this file shares a Unity build TU with
// test_realtime_profiler_sanity.cpp, which declares its own.
double stress_percentile(const std::vector<double>& sorted, double p) {
    const double rank = p * static_cast<double>(sorted.size() - 1);
    const auto lo = static_cast<size_t>(rank);
    const size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (rank - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

struct StressSyncErrorPercentilesUs {
    double p50 = 0.0;
    double p90 = 0.0;
    double p99 = 0.0;
    double max = 0.0;
};

// From one snapshot, so an in-progress report cannot mix percentiles taken at different instants.
StressSyncErrorPercentilesUs stress_sync_percentiles_us(const StressSyncErrorTally& tally) {
    const std::map<std::chrono::nanoseconds, uint64_t> counts = tally.counts();
    if (counts.empty()) {
        return {};
    }
    uint64_t total = 0;
    for (const auto& [value, count] : counts) {
        total += count;
    }
    // Value the sample at flattened position `rank` would have, without materializing the sorted list.
    const auto value_at_rank = [&counts](uint64_t rank) -> std::chrono::nanoseconds {
        uint64_t cumulative = 0;
        for (const auto& [value, count] : counts) {
            cumulative += count;
            if (rank < cumulative) {
                return value;
            }
        }
        return counts.rbegin()->first;
    };
    const auto pct_us = [&](double p) {
        const double rank = p * static_cast<double>(total - 1);
        const auto lo = static_cast<uint64_t>(rank);
        const std::chrono::nanoseconds lo_value = value_at_rank(lo);
        const std::chrono::nanoseconds hi_value = value_at_rank(std::min(lo + 1, total - 1));
        const auto interpolated = std::chrono::duration<double, std::nano>{lo_value} +
                                  (rank - static_cast<double>(lo)) * (hi_value - lo_value);
        return std::chrono::duration<double, std::micro>{interpolated}.count();
    };
    return {
        .p50 = pct_us(0.50),
        .p90 = pct_us(0.90),
        .p99 = pct_us(0.99),
        .max = std::chrono::duration<double, std::micro>{counts.rbegin()->first}.count(),
    };
}

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

distributed::MeshTraceId capture_blank_kernel_trace(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
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
    const distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < kNumProgramsInTrace; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);
    return trace_id;
}

std::shared_ptr<distributed::MeshDevice> open_unit_mesh() {
    return distributed::MeshDevice::create_unit_mesh(
        /*device_id=*/0, DEFAULT_L1_SMALL_SIZE, kTraceRegionSize, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
}

std::shared_ptr<distributed::MeshDevice> open_full_mesh() {
    return distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        kTraceRegionSize,
        1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
}

template <typename Opener>
std::shared_ptr<distributed::MeshDevice> open_profiler_mesh(Opener opener) {
    auto mesh_device = opener();
    if (mesh_device == nullptr) {
        return nullptr;
    }
    // RT profiler activation is decided at mesh open, so by the time the mesh is
    // opened this query is stable. When false, the dispatch config (ETH dispatch,
    // non-MMIO chip, kernels nullified, no valid RT core, worker_l1_size shrunk
    // below the ring size, ...) leaves RT profiler off; the test has nothing to
    // assert in that case so it skips cleanly.
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        return nullptr;
    }
    return mesh_device;
}

// The slope of the published device_cycle_offset against an independent host clock is the rate the published
// frequencies are systematically off from the chip's actual session-average rate -- every reported duration is
// divided by a published frequency, so a systematic bias puts the same fraction on every duration. Each record's
// frequency is its own probe-pair secant, so record-to-record scatter is expected and cancels here; what the
// regression catches is a bias common to all of them. Brief AICLK steps are far larger (~5200ppm) but average out
// over a run; this bounds what's left after they do. Measured under trace replay so the chip is hot.
TEST(RealtimeProfilerStress, FittedFrequencyTracksTheSessionAverage) {
    // 50ppm is 50ns on a 1ms program. Measured on Blackhole: under 10ppm.
    constexpr double kMaxSessionFrequencyErrorPpm = 50.0;
    constexpr size_t kMinAnchorsPerChip = 100;

    auto mesh_device = open_profiler_mesh(open_full_mesh);
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    struct Anchor {
        double host_ns;
        double offset;
    };
    std::mutex mutex;
    std::map<uint32_t, std::vector<Anchor>> anchors_by_chip;
    std::map<uint32_t, double> frequency_by_chip;
    // The mapping is anchored per record, so an offset that changed is no longer a re-anchor and cannot bound how much
    // is collected; at this record rate that is billions of samples. One per chip per millisecond is far more than a
    // 60s regression needs.
    constexpr auto kAnchorSampleInterval = std::chrono::milliseconds(1);
    std::map<uint32_t, std::chrono::steady_clock::time_point> last_sampled;
    const auto epoch = std::chrono::steady_clock::now();
    // device_cycle_offset is stated against the steady_clock epoch, which is boot, so it carries the published rate
    // times the machine's whole uptime. Regressed as-is, a rate that merely re-measures itself by a fraction of a ppm
    // moves the offset by uptime/window times as much as a real frequency error would -- 191833x on a box up 2.2 days,
    // and proportionally less on a freshly booted one, so the verdict would track uptime rather than the clock. Moving
    // the offsets onto an epoch-relative reference leaves the window itself as the only lever arm.
    const double epoch_host_ns = static_cast<double>(epoch.time_since_epoch().count());

    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        const auto now = std::chrono::steady_clock::now();
        const std::lock_guard lock(mutex);
        for (const auto& rec : batch.records) {
            const auto [it, inserted] = last_sampled.try_emplace(rec.chip_id, now);
            if (!inserted) {
                if (now - it->second < kAnchorSampleInterval) {
                    continue;
                }
                it->second = now;
            }
            const int64_t offset = rec.clock_sync.device_cycle_offset;
            frequency_by_chip[rec.chip_id] = rec.frequency;
            anchors_by_chip[rec.chip_id].push_back(
                {static_cast<double>((now - epoch).count()),
                 static_cast<double>(offset) + rec.frequency * epoch_host_ns});
        }
    });

    auto& cq = mesh_device->mesh_command_queue(0);
    const distributed::MeshTraceId trace_id = capture_blank_kernel_trace(mesh_device);

    const std::chrono::seconds window(
        tt::parse_env<std::uint32_t>("TT_RT_PROFILER_CLOCK_DRIFT_SECONDS", kDefaultClockDriftSeconds));
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < window) {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
    }
    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    UnregisterProgramRealtimeProfilerCallback(handle);

    double worst_ppm = 0.0;
    uint32_t worst_chip = 0;
    double min_ppm = 0.0;
    double max_ppm = 0.0;
    size_t chips_measured = 0;
    for (const auto& [chip, anchors] : anchors_by_chip) {
        if (anchors.size() < kMinAnchorsPerChip) {
            continue;
        }
        double mean_x = 0.0;
        double mean_y = 0.0;
        for (const auto& a : anchors) {
            mean_x += a.host_ns;
            mean_y += a.offset;
        }
        mean_x /= static_cast<double>(anchors.size());
        mean_y /= static_cast<double>(anchors.size());

        double num = 0.0;
        double den = 0.0;
        for (const auto& a : anchors) {
            const double dx = a.host_ns - mean_x;
            num += dx * (a.offset - mean_y);
            den += dx * dx;
        }
        const double ppm = (num / den) / frequency_by_chip[chip] * 1e6;  // slope (ticks/host-ns) as ppm of frequency
        min_ppm = chips_measured == 0 ? ppm : std::min(min_ppm, ppm);
        max_ppm = chips_measured == 0 ? ppm : std::max(max_ppm, ppm);
        if (std::abs(ppm) > std::abs(worst_ppm)) {
            worst_ppm = ppm;
            worst_chip = chip;
        }
        ++chips_measured;
    }

    ASSERT_GT(chips_measured, 0u) << "no device produced " << kMinAnchorsPerChip
                                  << " re-anchors; nothing to fit a drift rate from";
    log_info(
        tt::LogTest,
        "[RT profiler stress] fitted-frequency error over {}s across {} chip(s): min={:+.3f} ppm max={:+.3f} ppm | "
        "worst "
        "|error|={:+.3f} ppm on chip {} (limit {:.1f} ppm)",
        window.count(),
        chips_measured,
        min_ppm,
        max_ppm,
        worst_ppm,
        worst_chip,
        kMaxSessionFrequencyErrorPpm);

    EXPECT_LT(std::abs(worst_ppm), kMaxSessionFrequencyErrorPpm)
        << "chip " << worst_chip << " drifts at " << worst_ppm << " ppm, beyond the " << kMaxSessionFrequencyErrorPpm
        << " ppm a fitted frequency may be out by. Every reported duration is divided by that frequency and "
           "nothing re-anchors it, so past this every duration on this chip is wrong by the same fraction.";

    mesh_device->release_mesh_trace(trace_id);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerStress, PeakLoadPreservesRecords) {
    auto mesh_device = open_profiler_mesh(open_full_mesh);
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }
    auto* rt = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt, nullptr);
    const uint64_t num_active_devices = rt->num_active_devices();

    // Counted rather than asserted on: these run on the consumer's delivery thread, where a gtest assertion would
    // record its failure off the main thread.
    std::atomic<uint64_t> stress_records{0};
    std::atomic<uint64_t> inverted_timestamps{0};
    std::atomic<uint64_t> bad_frequency{0};
    std::atomic<uint64_t> implausible_duration{0};
    std::atomic<uint64_t> max_callback_batch{0};
    StressSyncErrorTally sync_errors;
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
            max_callback_batch.store(
                std::max<uint64_t>(max_callback_batch.load(std::memory_order_relaxed), batch.records.size()),
                std::memory_order_relaxed);
            for (const auto& rec : batch.records) {
                if (rec.runtime_id != kStressRuntimeId) {
                    continue;
                }
                stress_records.fetch_add(1, std::memory_order_relaxed);
                sync_errors.observe(rec);
                if (rec.end_timestamp < rec.start_timestamp) {
                    inverted_timestamps.fetch_add(1, std::memory_order_relaxed);
                }
                if (!(rec.frequency > 0.0)) {
                    bad_frequency.fetch_add(1, std::memory_order_relaxed);
                } else if (rec.duration().count() >= kMaxStressDurationNs) {
                    implausible_duration.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });

    auto& cq = mesh_device->mesh_command_queue(0);
    const distributed::MeshTraceId trace_id = capture_blank_kernel_trace(mesh_device);

    const std::chrono::seconds replay_window(
        tt::parse_env<std::uint32_t>("TT_RT_PROFILER_SATURATION_SECONDS", kDefaultStressReplaySeconds));
    uint64_t num_replays = 0;
    const auto replay_start = std::chrono::steady_clock::now();
    const auto replay_deadline = replay_start + replay_window;
    auto last_report = replay_start;
    for (;;) {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
        ++num_replays;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report >= kStressReportInterval) {
            const auto sync_us = stress_sync_percentiles_us(sync_errors);
            log_info(
                tt::LogTest,
                "[RT profiler stress] t={}s replays={} published={} peak_fifo={} this window, {} all-time of {} pages "
                "| sync error us: p50={:.2f} p90={:.2f} p99={:.2f} max={:.2f}",
                std::chrono::duration_cast<std::chrono::seconds>(now - replay_start).count(),
                num_replays,
                rt->num_published_records(),
                rt->take_peak_fifo_pages(),
                rt->peak_fifo_pages(),
                rt->host_fifo_capacity_pages(),
                sync_us.p50,
                sync_us.p90,
                sync_us.p99,
                sync_us.max);
            last_report = now;
        }
        if (now >= replay_deadline) {
            break;
        }
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    const uint32_t peak_fifo_pages = rt->peak_fifo_pages();
    const uint32_t fifo_capacity_pages = rt->host_fifo_capacity_pages();
    const uint32_t ring_full_waits = rt->read_ring_full_wait_count();
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(trace_id);

    const uint64_t expected_stress_records =
        static_cast<uint64_t>(kNumProgramsInTrace) * num_replays * num_active_devices;

    const auto sync_us = stress_sync_percentiles_us(sync_errors);
    log_info(
        tt::LogTest,
        "[RT profiler stress] {} stress records across {} active device(s) over {} replays, max_callback_batch={}, "
        "peak_fifo={}/{} pages, ring_full_waits={}, {} "
        "malformed-timestamp drops, {} bad-frequency, {} implausible-duration; "
        "sync error us: p50={:.2f} p90={:.2f} p99={:.2f} max={:.2f}",
        stress_records.load(),
        num_active_devices,
        num_replays,
        max_callback_batch.load(),
        peak_fifo_pages,
        fifo_capacity_pages,
        ring_full_waits,
        rt->num_malformed_records(),
        bad_frequency.load(),
        implausible_duration.load(),
        sync_us.p50,
        sync_us.p90,
        sync_us.p99,
        sync_us.max);

    ASSERT_GE(stress_records.load(), expected_stress_records)
        << "expected one record per program run: " << kNumProgramsInTrace << " programs per replay x " << num_replays
        << " replays x " << num_active_devices
        << " active device(s). A shortfall means profiler records were dropped at some point in the pipeline.";

    EXPECT_LT(peak_fifo_pages, fifo_capacity_pages)
        << "host D2H FIFO reached capacity; the receiver drained it slower than the device filled it";

    EXPECT_EQ(ring_full_waits, 0u)
        << "device ring reached capacity; the receiver drained it slower than the device filled it";

    // A record whose end precedes its start is a torn timestamp pair: BRISC NOC-reads start and end together out of
    // dispatch_s's L1, so any of them means that handoff let a read straddle two programs.
    EXPECT_EQ(rt->num_malformed_records(), 0u)
        << "the receiver discarded " << rt->num_malformed_records() << " record(s) with end_timestamp < "
        << "start_timestamp across " << stress_records.load()
        << " stress records; the dispatch_s -> BRISC timestamp handoff "
        << "is no longer atomic with respect to program boundaries.";
    // Distinct from num_malformed_records: the receiver drops these before publication, so a delivered one means
    // the filter itself regressed.
    EXPECT_EQ(inverted_timestamps.load(), 0u)
        << inverted_timestamps.load() << " delivered record(s) had end_timestamp < start_timestamp";
    EXPECT_EQ(bad_frequency.load(), 0u) << bad_frequency.load() << " stress record(s) had a non-positive frequency";
    EXPECT_EQ(implausible_duration.load(), 0u)
        << implausible_duration.load() << " stress record(s) reported duration >= " << kMaxStressDurationNs
        << " ns (clock corruption / mis-decoded timestamp)";

    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerStress, CallbackDeliveryLatency) {
    using namespace std::chrono_literals;
    constexpr uint32_t kPacedId = 0x6AC0;
    constexpr std::array<std::chrono::microseconds, 6> kGaps = {5us, 50us, 200us, 1000us, 5000us, 5us};
    constexpr uint32_t kOpsPerGap = 100;
    // Nothing gates publication: the receiver reads, probes, and publishes in one pass, so delivery latency is the
    // receiver's idle backoff (capped at 100us) plus the consumer wake. Measured worst per-gap p50 is ~74us and
    // p99 ~121us; the limits leave a few times that for CI noise while still catching a fixed oversleep anywhere
    // in the path.
    constexpr double kMaxPacedLatencyP50Us = 250.0;
    constexpr double kMaxPacedLatencyP99Us = 600.0;

    constexpr uint32_t num_gaps = static_cast<uint32_t>(kGaps.size());
    constexpr uint32_t total_paced = kOpsPerGap * num_gaps;

    auto mesh_device = open_profiler_mesh(open_unit_mesh);
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<std::atomic<std::chrono::steady_clock::time_point>> paced_delivered(total_paced);
    // A record flushes only when the NEXT program dispatches (device double-buffers the timestamps), so record k's
    // delivery latency is measured against host_start[k+1], the flush trigger.
    std::vector<std::atomic<std::chrono::steady_clock::time_point>> paced_host_start(total_paced);
    std::atomic<uint64_t> paced_idx{0};
    std::atomic<uint64_t> dropped_total{0};
    std::atomic<uint64_t> total_records{0};
    std::atomic<uint64_t> foreign_records{0};

    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
            dropped_total.fetch_add(batch.dropped, std::memory_order_relaxed);
            total_records.fetch_add(batch.records.size(), std::memory_order_relaxed);
            const auto now = std::chrono::steady_clock::now();
            for (const auto& rec : batch.records) {
                if (rec.runtime_id != kPacedId) {
                    foreign_records.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (rec.frequency > 0.0) {
                    const uint64_t idx = paced_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx < total_paced) {
                        paced_delivered[idx].store(now, std::memory_order_relaxed);
                        paced_host_start[idx].store(rec.host_start(), std::memory_order_relaxed);
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

    // Warm the consumer delivery thread with a few replays so its first-ever, slow-to-wake deliveries don't land in
    // the first measured bucket, then reset the counters so warm-up records don't shift the index pairing.
    constexpr uint32_t kWarmupOps = 32;
    for (uint32_t i = 0; i < kWarmupOps; ++i) {
        mesh_device->replay_mesh_trace(cq.id(), paced_trace, false);
    }
    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    paced_idx.store(0);
    total_records.store(0);
    foreign_records.store(0);
    dropped_total.store(0);

    for (uint32_t gap_idx = 0; gap_idx < num_gaps; ++gap_idx) {
        mesh_device->quiesce_devices();
        const auto gap = kGaps[gap_idx];
        const auto bucket_start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < kOpsPerGap; ++i) {
            while (std::chrono::steady_clock::now() < bucket_start + gap * i) {
            }
            mesh_device->replay_mesh_trace(cq.id(), paced_trace, false);
        }
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(paced_trace);

    const uint64_t paced_matched = std::min<uint64_t>(paced_idx.load(), total_paced);
    double worst_paced_latency_p50_us = 0.0;
    double worst_paced_latency_p99_us = 0.0;
    for (uint32_t gap_idx = 0; gap_idx < num_gaps; ++gap_idx) {
        std::vector<double> latency_us;
        latency_us.reserve(kOpsPerGap);
        // The last op in each bucket has no same-bucket successor (it flushes on the following quiesce), so it's
        // skipped.
        for (uint32_t i = 0; i + 1 < kOpsPerGap; ++i) {
            const uint32_t idx = gap_idx * kOpsPerGap + i;
            const auto d = paced_delivered[idx].load(std::memory_order_relaxed);
            const auto next_start = paced_host_start[idx + 1].load(std::memory_order_relaxed);
            constexpr std::chrono::steady_clock::time_point kUnset{};
            if (d == kUnset || next_start == kUnset) {
                continue;
            }
            latency_us.push_back(std::max(0.0, std::chrono::duration<double, std::micro>{d - next_start}.count()));
        }
        if (latency_us.empty()) {
            continue;
        }
        std::ranges::sort(latency_us);
        const double lat_p50 = stress_percentile(latency_us, 0.50);
        const double lat_p99 = stress_percentile(latency_us, 0.99);
        worst_paced_latency_p50_us = std::max(worst_paced_latency_p50_us, lat_p50);
        worst_paced_latency_p99_us = std::max(worst_paced_latency_p99_us, lat_p99);
        log_info(
            tt::LogTest,
            "[RT profiler stress] gap={:5}us | delivery latency (available->callback) p50={:.1f} p99={:.1f} "
            "max={:.1f}us",
            kGaps[gap_idx].count(),
            lat_p50,
            lat_p99,
            latency_us.back());
    }

    log_info(
        tt::LogTest,
        "[RT profiler stress] callback delivery: total_records={} paced={} foreign={} dropped={}",
        total_records.load(),
        paced_idx.load(),
        foreign_records.load(),
        dropped_total.load());

    EXPECT_EQ(dropped_total.load(), 0u)
        << "callback dropped records; deliveries are paired to enqueues by position, so a drop misaligns every "
        << "later pair and the latencies below are meaningless";
    EXPECT_GE(paced_matched, total_paced - total_paced / 100)
        << "too few of the " << total_paced << " paced ops reached the callback; the latency percentiles "
        << "below are over a partial sample and unreliable";
    EXPECT_LT(worst_paced_latency_p50_us, kMaxPacedLatencyP50Us)
        << "median delivery latency exceeds the staging wait by too much; the consumer is not waking promptly (a "
        << "fixed backoff/oversleep would show up here)";
    EXPECT_LT(worst_paced_latency_p99_us, kMaxPacedLatencyP99Us)
        << "tail delivery latency too high; occasional long stalls in the delivery path";
    EXPECT_TRUE(mesh_device->close());
}

// Three consumers read the same record stream at different throttled rates. Verifies the per-reader
// drop accounting: for every consumer, received + dropped covers every record produced, and a
// throttled consumer drops no more than its sustain rate forces (no over-dropping).
TEST(RealtimeProfilerStress, ConsumerDropAccountingUnderLoad) {
    const std::chrono::seconds run_window(
        tt::parse_env<std::uint32_t>("TT_RT_PROFILER_DROP_ACCOUNTING", kDefaultDropAccountingSeconds));

    auto mesh_device = open_profiler_mesh(open_full_mesh);
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }
    const uint64_t num_devices = mesh_device->num_devices();
    auto* rt = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt, nullptr);

    auto& cq = mesh_device->mesh_command_queue(0);
    const distributed::MeshTraceId trace_id = capture_blank_kernel_trace(mesh_device);

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
    // A reader sees only what is published after it is made, so the pipeline has to be empty before the three
    // registrations below: calibration records still draining would land in whichever readers already existed and make
    // the cross-consumer totals disagree by however many slipped in between two RegisterProgramRealtimeProfilerCallback
    // calls. quiesce_devices() stops the devices but does not wait for the profiler to drain.
    std::this_thread::sleep_for(kPostQuiesceDrain);

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
    StressSyncErrorTally sync_errors;

    auto make_consumer = [](Counters& c, std::chrono::nanoseconds per_record, StressSyncErrorTally* samples = nullptr) {
        return [&c, per_record, samples, start = std::chrono::steady_clock::time_point{}, paced = uint64_t{0}](
                   const ProgramRealtimeRecordBatch& batch) mutable {
            c.received.fetch_add(batch.records.size(), std::memory_order_relaxed);
            c.dropped.fetch_add(batch.dropped, std::memory_order_relaxed);
            if (samples != nullptr) {
                for (const auto& rec : batch.records) {
                    samples->observe(rec);
                }
            }
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

    ProgramRealtimeProfilerCallbackHandle h_keeps_up = RegisterProgramRealtimeProfilerCallback(
        make_consumer(keeps_up, std::chrono::nanoseconds::zero(), &sync_errors));
    ProgramRealtimeProfilerCallbackHandle h_borderline =
        RegisterProgramRealtimeProfilerCallback(make_consumer(borderline, borderline_per_record));
    ProgramRealtimeProfilerCallbackHandle h_slow =
        RegisterProgramRealtimeProfilerCallback(make_consumer(slow, slow_per_record));

    const auto run_start = std::chrono::steady_clock::now();
    const auto run_deadline = run_start + run_window;
    auto last_report = run_start;
    for (;;) {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report >= kStressReportInterval) {
            const auto sync_us = stress_sync_percentiles_us(sync_errors);
            log_info(
                tt::LogTest,
                "[RT profiler stress] t={}s keeps_up: recv={} drop={} | borderline: recv={} drop={} | slow: recv={} "
                "drop={} | peak_fifo={} this window, {} all-time of {} pages | sync error us: "
                "p50={:.2f} p90={:.2f} p99={:.2f} max={:.2f}",
                std::chrono::duration_cast<std::chrono::seconds>(now - run_start).count(),
                keeps_up.received.load(),
                keeps_up.dropped.load(),
                borderline.received.load(),
                borderline.dropped.load(),
                slow.received.load(),
                slow.dropped.load(),
                rt->take_peak_fifo_pages(),
                rt->peak_fifo_pages(),
                rt->host_fifo_capacity_pages(),
                sync_us.p50,
                sync_us.p90,
                sync_us.p99,
                sync_us.max);
            last_report = now;
        }
        if (now >= run_deadline) {
            break;
        }
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);

    const uint32_t peak_fifo_pages = rt->peak_fifo_pages();
    const uint32_t fifo_capacity_pages = rt->host_fifo_capacity_pages();

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

    const auto sync_us = stress_sync_percentiles_us(sync_errors);
    log_info(
        tt::LogTest,
        "[RT profiler stress] devices={} total={} peak_fifo={}/{} pages | "
        "borderline: recv={} drop={} sum={} | slow: recv={} drop={} sum={} | "
        "sync error us: p50={:.2f} p90={:.2f} p99={:.2f} max={:.2f}",
        num_devices,
        keeps_up_received,
        peak_fifo_pages,
        fifo_capacity_pages,
        borderline_received,
        borderline_dropped,
        borderline_total,
        slow_received,
        slow_dropped,
        slow_total,
        sync_us.p50,
        sync_us.p90,
        sync_us.p99,
        sync_us.max);

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
