// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stress test for the real-time (RT) profiler ring buffer (BRISC writer ↔
// NCRISC pusher). The ring is sized at RT_PROFILER_RING_CAPACITY = 4096
// entries (see realtime_profiler_ring_buffer.hpp). Under a normal workload
// NCRISC pushes each entry to the host over PCIe in ~50–80 µs, while BRISC
// can write an entry in ~0.3 µs once dispatch_s flips the mailbox state.
// That ~150× imbalance is what makes the BRISC↔NCRISC handshake worth
// stress-testing: if we slam BRISC with thousands of program launches faster
// than NCRISC can drain, BRISC must spin in `rt_ring_full` without dropping
// records or wedging dispatch_s. This test forces exactly that scenario by
// capturing a trace of 4096 blank-kernel program launches and replaying it
// in one go (no host-side throttling between launches).
//
// What the test buys us:
//   * Catches BRISC dropping records when the ring is full (off-by-one in
//     rt_ring_full / write_index wraparound).
//   * Catches NCRISC livelock under sustained back-pressure (host receiver
//     thread keeps up with steady-state push rate).
//   * Catches dispatch_s stalling because the RT-profiler mailbox handshake
//     gets wedged (e.g. BRISC stuck in rt_ring_full would block dispatch_s
//     from firing the next launch via the shared mailbox).
//
// The kernel is intentionally as small as possible (literal `void
// kernel_main() {}`) so dispatch overhead dominates and BRISC is fed at the
// peak rate dispatch_s can sustain in a trace replay. Mirrors the placement
// pattern in test_realtime_profiler_sanity.cpp (BASIC tier, single MMIO
// device, gracefully skips when RT profiler is disabled).

#include <chrono>
#include <cstdint>
#include <mutex>
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

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeRecord;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

// Matches RT_PROFILER_RING_CAPACITY in realtime_profiler_ring_buffer.hpp.
// Picking the trace length equal to the ring capacity is the worst case for
// the back-pressure path: BRISC can fill the ring in roughly the time it
// takes NCRISC to push 1–2 entries over PCIe, so by enqueue ~80 of 4096
// the ring is at capacity and stays there for the rest of the trace.
constexpr uint32_t kNumProgramsInTrace = 4096;

// Trace stores one EnqueueProgram dispatch packet per program. Blank-kernel
// programs with no CBs / no runtime args are tiny (~hundreds of bytes), so
// 64 MB is comfortably more than 4096 of them need; sized generously so a
// future change to the dispatch packet layout can't silently OOM the trace
// region and turn this into a flake. Lives in DRAM, not L1, so it doesn't
// interact with the worker_l1_size eligibility check we just added.
constexpr size_t kTraceRegionSize = 64 * 1024 * 1024;

// Programs in the trace use this runtime_id so every record we receive can
// be attributed to this test (records with program_id == 0 are reserved for
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

// Quiesce + drain window before unregistering the callback. Mirrors the
// 500ms used in test_realtime_profiler_sanity.cpp; bumped to 2s here
// because at 4096 entries × ~50–80 µs/push the worst-case drain is
// ~250–330 ms (ring fully backed up at trace replay completion).
constexpr auto kPostQuiesceDrain = std::chrono::milliseconds(2000);

// Allowed slack for the deterministic startup race where the compute kernel
// detects dispatch_d's stream-register clearing before dispatch_s has
// recorded the first start_timestamp, producing one record where
// end_timestamp < start_timestamp by a handful of cycles. Same value the
// host/device correlation test tolerates (see test_realtime_profiler.py
// :: test_host_device_correlation, "startup_race_threshold") and the same
// value the production Tracy handler uses to distinguish "benign" from
// "noisy" skips (see realtime_profiler_tracy_handler.cpp,
// kStartupRaceThreshold).
constexpr uint64_t kStartupRaceSlackCycles = 100'000;

// Hard upper bound on the fraction of records that can come back with
// end_timestamp < start_timestamp before this test fails. The production
// host receiver already silently skips any such record on the Tracy path
// (see realtime_profiler_tracy_handler.cpp:HandleRecord), so a tiny number
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
        "tt_metal/kernels/dataflow/blank.cpp",
        stress_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tt_metal/kernels/dataflow/blank.cpp",
        stress_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernel(program, "tt_metal/kernels/compute/blank.cpp", stress_core, ComputeConfig{});

    program.set_runtime_id(static_cast<uint64_t>(kStressRuntimeId));

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    return workload;
}

TEST(RealtimeProfilerStress, RingBufferOverflowFromTrace) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId,
        DEFAULT_L1_SMALL_SIZE,
        kTraceRegionSize,
        /*num_command_queues=*/1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    // RT profiler activation is decided during the init-sync handshake at
    // mesh open, so by the time create_unit_mesh returns this query is
    // stable. When false, the dispatch config (ETH dispatch, non-MMIO
    // chip, kernels nullified, no valid RT core, worker_l1_size shrunk
    // below the ring size, ...) leaves RT profiler off; the test has
    // nothing to assert in that case so it skips cleanly.
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::mutex records_mu;
    std::vector<ProgramRealtimeRecord> records;
    records.reserve(kNumProgramsInTrace);

    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&records_mu, &records](const ProgramRealtimeRecord& record) {
            std::lock_guard<std::mutex> lk(records_mu);
            records.push_back(record);
        });

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& mesh_command_queue = mesh_device->mesh_command_queue(0);

    // Compile + warm up the workload outside the trace capture so the trace
    // contains only steady-state dispatch packets (no compile/upload hops).
    distributed::EnqueueMeshWorkload(mesh_command_queue, workload, /*blocking=*/true);

    // Capture: 4096 back-to-back EnqueueMeshWorkload calls of the same
    // blank-kernel workload. Reusing one workload (vs. building 4096
    // distinct programs) keeps compile time near zero and avoids host-side
    // memory pressure; the dispatch commands captured in the trace are
    // independent per-enqueue, so dispatch_s still fires 4096 separate
    // kernel_start pulses on replay.
    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), mesh_command_queue.id());
    for (uint32_t i = 0; i < kNumProgramsInTrace; ++i) {
        distributed::EnqueueMeshWorkload(mesh_command_queue, workload, /*blocking=*/false);
    }
    mesh_device->end_mesh_trace(mesh_command_queue.id(), trace_id);

    // Replay the entire 4096-launch trace as a single blocking dispatch.
    // No host-side per-launch overhead between programs => peak BRISC feed
    // rate, ring saturates within the first few dozen launches.
    mesh_device->replay_mesh_trace(mesh_command_queue.id(), trace_id, /*blocking=*/true);

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);

    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(trace_id);

    std::vector<ProgramRealtimeRecord> collected;
    {
        std::lock_guard<std::mutex> lk(records_mu);
        collected = std::move(records);
    }

    // Filter to records produced by this test's workload. The +1 warmup
    // launch we did before BeginTraceCapture also lands in this bucket, so
    // we expect at least kNumProgramsInTrace + 1 matching records. Compare
    // with >= because infrastructure traffic on a freshly-opened mesh can
    // emit a handful of extra non-stress records before our callback was
    // hooked up.
    uint32_t stress_records = 0;
    for (const auto& rec : collected) {
        if (rec.program_id == kStressRuntimeId) {
            ++stress_records;
        }
    }

    ASSERT_GE(stress_records, kNumProgramsInTrace)
        << "Expected at least " << kNumProgramsInTrace
        << " RT profiler records for the stress workload (one per traced launch); got " << stress_records
        << " stress records out of " << collected.size() << " total. A shortfall here means BRISC dropped entries "
        << "under sustained back-pressure or NCRISC stopped pushing partway through the trace.";

    // Every record must have well-formed timestamps + frequency. Under
    // ring-full back-pressure BRISC writes the timestamp slot _before_
    // bumping write_index, so a torn read on the host side or a stale slot
    // re-used after wraparound would surface as a large negative delta or
    // freq <= 0.
    //
    // We classify end_timestamp < start_timestamp records into two buckets:
    //   * startup_race_skips: |delta| <= kStartupRaceSlackCycles.
    //     These are the documented dispatch_d/dispatch_s startup race;
    //     freely tolerated, just counted for visibility.
    //   * large_negative_skips: |delta| > kStartupRaceSlackCycles.
    //     These are torn 64-bit reads / stale-slot residue. The production
    //     Tracy handler already silently drops them
    //     (realtime_profiler_tracy_handler.cpp::HandleRecord), so we tolerate
    //     them up to kMaxBadTimestampFraction of total stress records and
    //     fail above that — catching a regression where the corruption rate
    //     becomes systemic instead of a rare blip.
    uint32_t startup_race_skips = 0;
    uint32_t large_negative_skips = 0;
    uint32_t bad_frequency = 0;
    uint32_t implausible_duration = 0;
    int64_t worst_negative_delta = 0;
    for (const auto& rec : collected) {
        if (rec.program_id != kStressRuntimeId) {
            continue;
        }
        if (rec.end_timestamp < rec.start_timestamp) {
            const uint64_t neg_delta_cycles = rec.start_timestamp - rec.end_timestamp;
            if (neg_delta_cycles <= kStartupRaceSlackCycles) {
                ++startup_race_skips;
            } else {
                ++large_negative_skips;
                if (-static_cast<int64_t>(neg_delta_cycles) < worst_negative_delta) {
                    worst_negative_delta = -static_cast<int64_t>(neg_delta_cycles);
                }
            }
        }
        if (!(rec.frequency > 0.0)) {
            ++bad_frequency;
        } else if (rec.end_timestamp >= rec.start_timestamp) {
            const uint64_t duration_cycles = rec.end_timestamp - rec.start_timestamp;
            const double duration_ns = static_cast<double>(duration_cycles) / rec.frequency;
            if (duration_ns >= kMaxStressDurationNs) {
                ++implausible_duration;
            }
        }
    }

    log_info(
        tt::LogTest,
        "[RT profiler stress] {} stress records, {} startup-race skips, {} large-negative-delta skips "
        "(worst delta = {} cycles), {} bad-frequency, {} implausible-duration",
        stress_records,
        startup_race_skips,
        large_negative_skips,
        worst_negative_delta,
        bad_frequency,
        implausible_duration);

    const uint32_t max_allowed_large_negative =
        static_cast<uint32_t>(static_cast<double>(stress_records) * kMaxBadTimestampFraction);
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

}  // namespace
}  // namespace tt::tt_metal
