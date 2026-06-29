// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include <sys/resource.h>
#include <unistd.h>

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

#include "tt_metal/distributed/realtime_profiler_manager.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeRecord;
using tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
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

// Quiesce + drain window before unregistering the callback. Mirrors the
// 500ms used in test_realtime_profiler_sanity.cpp; bumped to 2s here
// because at 4096 entries × ~50–80 µs/push the worst-case drain is
// ~250–330 ms (ring fully backed up at trace replay completion).
constexpr auto kPostQuiesceDrain = std::chrono::milliseconds(2000);

namespace SustainedStressConfig {
constexpr const char* kSecondsEnv = "TT_RT_PROFILER_STRESS_SECONDS";
constexpr const char* kPrintSecondsEnv = "TT_RT_PROFILER_STRESS_PRINT_SEC";
constexpr const char* kCallbacksEnv = "TT_RT_PROFILER_STRESS_CALLBACKS";
constexpr const char* kSlowCallbackUsEnv = "TT_RT_PROFILER_STRESS_SLOW_CALLBACK_US";
constexpr const char* kExpectCallbackDropsEnv = "TT_RT_PROFILER_STRESS_EXPECT_CALLBACK_DROPS";
constexpr double kDefaultPrintSeconds = 10.0;
}  // namespace SustainedStressConfig

namespace BurstyStressConfig {
constexpr const char* kSecondsEnv = "TT_RT_PROFILER_BURSTY_STRESS_SECONDS";
constexpr const char* kPrintSecondsEnv = "TT_RT_PROFILER_BURSTY_STRESS_PRINT_SEC";
constexpr const char* kCallbacksEnv = "TT_RT_PROFILER_BURSTY_STRESS_CALLBACKS";
constexpr const char* kExpectCallbackDropsEnv = "TT_RT_PROFILER_BURSTY_STRESS_EXPECT_CALLBACK_DROPS";
constexpr const char* kProgramsPerBurstEnv = "TT_RT_PROFILER_BURSTY_STRESS_PROGRAMS_PER_BURST";
constexpr const char* kGapProfileEnv = "TT_RT_PROFILER_BURSTY_STRESS_GAP_PROFILE";
constexpr double kDefaultPrintSeconds = 10.0;
constexpr uint32_t kProgramsPerBurst = 128;
constexpr uint32_t kIdleBackoffMicros = 1000;
inline constexpr std::array<uint32_t, 7> kGapMicros = {kIdleBackoffMicros, 1050, 1100, 1150, 1250, 1500, 2000};
inline constexpr std::array<uint32_t, 9> kSpinGapMicros = {0, 5, 10, 20, 40, 80, 160, 320, kIdleBackoffMicros};
}  // namespace BurstyStressConfig

namespace IdleCpuConfig {
constexpr const char* kSecondsEnv = "TT_RT_PROFILER_IDLE_CPU_SECONDS";
constexpr const char* kCallbacksEnv = "TT_RT_PROFILER_IDLE_CPU_CALLBACKS";
constexpr const char* kWarmupMillisEnv = "TT_RT_PROFILER_IDLE_CPU_WARMUP_MS";
constexpr auto kDefaultWarmup = std::chrono::milliseconds(500);
}  // namespace IdleCpuConfig

namespace MixedStressConfig {
constexpr const char* kSecondsEnv = "TT_RT_PROFILER_MIXED_STRESS_SECONDS";
constexpr const char* kPrintSecondsEnv = "TT_RT_PROFILER_MIXED_STRESS_PRINT_SEC";
constexpr const char* kCallbacksEnv = "TT_RT_PROFILER_MIXED_STRESS_CALLBACKS";
constexpr const char* kSlowNsEnv = "TT_RT_PROFILER_MIXED_STRESS_SLOW_NS";
constexpr const char* kSpikeSecEnv = "TT_RT_PROFILER_MIXED_STRESS_SPIKE_SEC";
constexpr double kDefaultPrintSeconds = 10.0;
constexpr uint32_t kDefaultCallbacks = 9;
constexpr uint32_t kDefaultSlowNs = 60;
constexpr double kDefaultSpikeSeconds = 5.0;
}  // namespace MixedStressConfig

bool env_flag_enabled(const char* value) {
    if (value == nullptr) {
        return false;
    }
    const std::string_view v(value);
    return v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON" || v == "yes" || v == "YES";
}

double process_cpu_seconds() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
    return static_cast<double>(usage.ru_utime.tv_sec) + static_cast<double>(usage.ru_utime.tv_usec) / 1'000'000.0 +
           static_cast<double>(usage.ru_stime.tv_sec) + static_cast<double>(usage.ru_stime.tv_usec) / 1'000'000.0;
}

std::unordered_map<int, double> thread_cpu_seconds_by_tid() {
    std::unordered_map<int, double> result;
    const long ticks_per_second = sysconf(_SC_CLK_TCK);
    if (ticks_per_second <= 0) {
        return result;
    }

    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
        const std::string tid_string = entry.path().filename().string();
        int tid = 0;
        try {
            tid = std::stoi(tid_string);
        } catch (...) {
            continue;
        }

        std::ifstream stat_file(entry.path() / "stat");
        std::string stat_line;
        if (!std::getline(stat_file, stat_line)) {
            continue;
        }
        const size_t end_comm = stat_line.rfind(')');
        if (end_comm == std::string::npos || end_comm + 2 >= stat_line.size()) {
            continue;
        }

        std::istringstream fields(stat_line.substr(end_comm + 2));
        std::string field;
        uint64_t utime_ticks = 0;
        uint64_t stime_ticks = 0;
        for (uint32_t index = 0; fields >> field; ++index) {
            if (index == 11) {
                utime_ticks = std::strtoull(field.c_str(), nullptr, 10);
            } else if (index == 12) {
                stime_ticks = std::strtoull(field.c_str(), nullptr, 10);
                break;
            }
        }
        result.emplace(tid, static_cast<double>(utime_ticks + stime_ticks) / static_cast<double>(ticks_per_second));
    }
    return result;
}

std::unordered_map<int, std::string> thread_comm_by_tid() {
    std::unordered_map<int, std::string> result;
    for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task")) {
        int tid = 0;
        try {
            tid = std::stoi(entry.path().filename().string());
        } catch (...) {
            continue;
        }
        std::ifstream comm_file(entry.path() / "comm");
        std::string comm;
        if (std::getline(comm_file, comm)) {
            result.emplace(tid, comm);
        }
    }
    return result;
}

void update_max(std::atomic<uint64_t>& target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value && !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

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

    std::vector<ProgramRealtimeRecord> records;
    records.reserve(kNumProgramsInTrace);

    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&records](const ProgramRealtimeRecordBatch& batch) {
            records.insert(records.end(), batch.records.begin(), batch.records.end());
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

    std::vector<ProgramRealtimeRecord> collected = std::move(records);

    // Filter to records produced by this test's workload. The +1 warmup
    // launch we did before BeginTraceCapture also lands in this bucket, so
    // we expect at least kNumProgramsInTrace + 1 matching records. Compare
    // with >= because infrastructure traffic on a freshly-opened mesh can
    // emit a handful of extra non-stress records before our callback was
    // hooked up.
    uint32_t stress_records = 0;
    for (const auto& rec : collected) {
        if (rec.runtime_id == kStressRuntimeId) {
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
        if (rec.runtime_id != kStressRuntimeId) {
            continue;
        }
        if (rec.end_timestamp < rec.start_timestamp) {
            const uint64_t neg_delta_cycles = rec.start_timestamp - rec.end_timestamp;
            if (neg_delta_cycles <= kStartupRaceSlackCycles) {
                ++startup_race_skips;
            } else {
                ++large_negative_skips;
                worst_negative_delta = std::min(-static_cast<int64_t>(neg_delta_cycles), worst_negative_delta);
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

// Sustained-throughput RT profiler stress, opt-in via TT_RT_PROFILER_STRESS_SECONDS.
// Opens the full mesh and replays a blank-kernel trace for the requested duration.
//
//   TT_RT_PROFILER_STRESS_SECONDS=600 ./<binary> --gtest_filter='*HostFifoPressureSustained'
//   TT_RT_PROFILER_STRESS_PRINT_SEC=5  (optional; default 10s between prints)
//   TT_RT_PROFILER_STRESS_CALLBACKS=8  (optional; default 1 - each gets its own consumer thread)
//   TT_RT_PROFILER_STRESS_SLOW_CALLBACK_US=1000 TT_RT_PROFILER_STRESS_EXPECT_CALLBACK_DROPS=1
TEST(RealtimeProfilerStress, HostFifoPressureSustained) {
    const char* secs_env = std::getenv(SustainedStressConfig::kSecondsEnv);
    if (secs_env == nullptr) {
        GTEST_SKIP() << "Set " << SustainedStressConfig::kSecondsEnv
                     << "=<n> to run the sustained host-FIFO pressure stress";
    }
    const double run_seconds = std::atof(secs_env);
    if (run_seconds <= 0.0) {
        GTEST_SKIP() << SustainedStressConfig::kSecondsEnv << " must be positive";
    }
    const char* print_env = std::getenv(SustainedStressConfig::kPrintSecondsEnv);
    const double print_seconds =
        (print_env != nullptr) ? std::max(0.1, std::atof(print_env)) : SustainedStressConfig::kDefaultPrintSeconds;
    const char* slow_callback_env = std::getenv(SustainedStressConfig::kSlowCallbackUsEnv);
    const uint32_t slow_callback_us =
        (slow_callback_env != nullptr) ? static_cast<uint32_t>(std::max(0, std::atoi(slow_callback_env))) : 0;
    const bool expect_callback_drops = env_flag_enabled(std::getenv(SustainedStressConfig::kExpectCallbackDropsEnv));
    const char* callbacks_env = std::getenv(SustainedStressConfig::kCallbacksEnv);
    const uint32_t num_callbacks =
        (callbacks_env != nullptr) ? static_cast<uint32_t>(std::max(1, std::atoi(callbacks_env))) : 1;

    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        kTraceRegionSize,
        1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    // One counter pair per callback; each callback reads the full record stream on its own
    // consumer thread, so per-callback counters keep drop attribution distinct.
    struct CallbackCounters {
        std::atomic<uint64_t> records{0};
        std::atomic<uint64_t> dropped{0};
    };
    std::vector<std::unique_ptr<CallbackCounters>> counters;
    std::vector<ProgramRealtimeProfilerCallbackHandle> handles;
    counters.reserve(num_callbacks);
    handles.reserve(num_callbacks);
    for (uint32_t i = 0; i < num_callbacks; ++i) {
        counters.push_back(std::make_unique<CallbackCounters>());
        CallbackCounters* c = counters.back().get();
        handles.push_back(
            RegisterProgramRealtimeProfilerCallback([c, slow_callback_us](const ProgramRealtimeRecordBatch& batch) {
                c->records.fetch_add(batch.records.size(), std::memory_order_relaxed);
                c->dropped.fetch_add(batch.dropped, std::memory_order_relaxed);
                if (slow_callback_us != 0) {
                    std::this_thread::sleep_for(std::chrono::microseconds(slow_callback_us));
                }
            }));
    }

    auto sum_records = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->records.load(std::memory_order_relaxed);
        }
        return sum;
    };
    auto sum_dropped = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->dropped.load(std::memory_order_relaxed);
        }
        return sum;
    };

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(cq, workload, true);  // compile + warm up

    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < kNumProgramsInTrace; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    std::atomic<bool> stop{false};
    std::thread monitor([&]() {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t last_records = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::duration<double>(print_seconds));
            const uint64_t total = sum_records();
            const uint64_t delta = total - last_records;
            last_records = total;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const auto* rt = mesh_device->impl().get_realtime_profiler();
            const uint64_t pub_records = rt ? rt->num_published_records() : 0;
            const uint64_t pub_batches = rt ? rt->num_published_batches() : 0;
            const double mean_publish_batch = pub_batches ? static_cast<double>(pub_records) / pub_batches : 0.0;
            log_info(
                tt::LogTest,
                "[RT stress t={:6.1f}s] records +{} ({:.0f}/s summed over {} callbacks) total={} | "
                "max_fifo={} pages, mean_publish_batch={:.1f} records, callback_dropped={}",
                elapsed,
                delta,
                static_cast<double>(delta) / print_seconds,
                num_callbacks,
                total,
                rt ? rt->peak_fifo_pages() : 0u,
                mean_publish_batch,
                sum_dropped());
        }
    });

    const auto t_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() < run_seconds) {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    stop.store(true, std::memory_order_relaxed);
    monitor.join();

    const auto* final_rt = mesh_device->impl().get_realtime_profiler();
    const uint64_t final_pub_records = final_rt ? final_rt->num_published_records() : 0;
    const uint64_t final_pub_batches = final_rt ? final_rt->num_published_batches() : 0;
    const double final_mean_publish_batch =
        final_pub_batches ? static_cast<double>(final_pub_records) / final_pub_batches : 0.0;
    log_info(
        tt::LogTest,
        "[RT stress FINAL] callbacks={} total_records={} (summed over callbacks) max_fifo={} pages, "
        "mean_publish_batch={:.1f} records, callback_dropped={}",
        num_callbacks,
        sum_records(),
        final_rt ? final_rt->peak_fifo_pages() : 0u,
        final_mean_publish_batch,
        sum_dropped());

    for (auto handle : handles) {
        UnregisterProgramRealtimeProfilerCallback(handle);
    }
    mesh_device->release_mesh_trace(trace_id);

    EXPECT_GT(sum_records(), 0u) << "No RT profiler records received during the stress run";
    if (expect_callback_drops) {
        EXPECT_GT(sum_dropped(), 0u) << "Expected callback drops; increase "
                                     << SustainedStressConfig::kSlowCallbackUsEnv << " or run duration";
    }
    EXPECT_TRUE(mesh_device->close());
}

// Mixed-behavior callback stress: registers FAST, SLOW, and SPIKY consumers at once
// (round-robin by index % 3) so the ring is driven simultaneously by readers that keep
// up, readers that steadily drop, and readers that oscillate between the two. A modestly
// slow reader that hovers at the drop threshold is harder on the ring than a very slow
// one (which just idles after dropping), and the spiky readers force the writer's
// drop-on-lag / recovery path to engage and disengage repeatedly. The per-behavior
// record/drop counts are logged; eyeball that fast stays at ~0 drops, slow drops
// steadily, and spiky's drops track its spike phase.
//
//   TT_RT_PROFILER_MIXED_STRESS_SECONDS=300 ./<binary> --gtest_filter='*MixedCallbackBehaviors'
//   TT_RT_PROFILER_MIXED_STRESS_PRINT_SEC=3   (optional; default 10s between prints)
//   TT_RT_PROFILER_MIXED_STRESS_CALLBACKS=9   (optional; default 9; split fast/slow/spiky by index % 3)
//   TT_RT_PROFILER_MIXED_STRESS_SLOW_NS=60    (optional; simulated per-record cost in ns; >~40 at 25M/s -> drops)
//   TT_RT_PROFILER_MIXED_STRESS_SPIKE_SEC=5   (optional; spiky readers flip keep-up/fall-behind each period)
TEST(RealtimeProfilerStress, MixedCallbackBehaviors) {
    const char* secs_env = std::getenv(MixedStressConfig::kSecondsEnv);
    if (secs_env == nullptr) {
        GTEST_SKIP() << "Set " << MixedStressConfig::kSecondsEnv << "=<n> to run the mixed-callback-behavior stress";
    }
    const double run_seconds = std::atof(secs_env);
    if (run_seconds <= 0.0) {
        GTEST_SKIP() << MixedStressConfig::kSecondsEnv << " must be positive";
    }
    const char* print_env = std::getenv(MixedStressConfig::kPrintSecondsEnv);
    const double print_seconds =
        (print_env != nullptr) ? std::max(0.1, std::atof(print_env)) : MixedStressConfig::kDefaultPrintSeconds;
    const char* callbacks_env = std::getenv(MixedStressConfig::kCallbacksEnv);
    const uint32_t num_callbacks = (callbacks_env != nullptr)
                                       ? static_cast<uint32_t>(std::max(0, std::atoi(callbacks_env)))
                                       : MixedStressConfig::kDefaultCallbacks;
    const char* slow_env = std::getenv(MixedStressConfig::kSlowNsEnv);
    const uint32_t slow_ns = (slow_env != nullptr) ? static_cast<uint32_t>(std::max(0, std::atoi(slow_env)))
                                                   : MixedStressConfig::kDefaultSlowNs;
    const char* spike_env = std::getenv(MixedStressConfig::kSpikeSecEnv);
    const double spike_period =
        (spike_env != nullptr) ? std::max(0.1, std::atof(spike_env)) : MixedStressConfig::kDefaultSpikeSeconds;

    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        kTraceRegionSize,
        1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    enum class Behavior : uint8_t { Fast, Slow, Spiky };
    struct CallbackCounters {
        std::atomic<uint64_t> records{0};
        std::atomic<uint64_t> dropped{0};
        Behavior behavior{Behavior::Fast};
    };

    // Spiky readers derive their fast/slow phase from this shared anchor, so they all
    // fall behind and recover together (synchronized swings are harder on the ring).
    const auto spike_t0 = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<CallbackCounters>> counters;
    std::vector<ProgramRealtimeProfilerCallbackHandle> handles;
    counters.reserve(num_callbacks);
    handles.reserve(num_callbacks);
    for (uint32_t i = 0; i < num_callbacks; ++i) {
        const Behavior behavior = static_cast<Behavior>(i % 3);
        counters.push_back(std::make_unique<CallbackCounters>());
        CallbackCounters* c = counters.back().get();
        c->behavior = behavior;
        handles.push_back(RegisterProgramRealtimeProfilerCallback(
            [c, behavior, slow_ns, spike_period, spike_t0](const ProgramRealtimeRecordBatch& batch) {
                c->records.fetch_add(batch.records.size(), std::memory_order_relaxed);
                c->dropped.fetch_add(batch.dropped, std::memory_order_relaxed);
                bool throttle = false;
                switch (behavior) {
                    case Behavior::Fast: break;
                    case Behavior::Slow: throttle = true; break;
                    case Behavior::Spiky: {
                        const double elapsed =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - spike_t0).count();
                        throttle = (static_cast<long long>(elapsed / spike_period) % 2) == 1;
                        break;
                    }
                }
                // Simulate a per-record processing cost. A fixed per-batch sleep is useless here: a reader
                // that falls behind just reads bigger batches (up to the consumer batch cap), so
                // the fixed cost is amortized away and it catches right back up. Scaling the sleep by batch
                // size throttles the drain to ~1/slow_ns records/s regardless of batch size, so the reader
                // actually lags and drops once slow_ns exceeds the per-record publish budget (~40ns at 25M/s).
                if (throttle && slow_ns != 0) {
                    std::this_thread::sleep_for(std::chrono::nanoseconds(batch.records.size() * slow_ns));
                }
            }));
    }

    auto group_records = [&counters](Behavior b) {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            if (c->behavior == b) {
                sum += c->records.load(std::memory_order_relaxed);
            }
        }
        return sum;
    };
    auto group_dropped = [&counters](Behavior b) {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            if (c->behavior == b) {
                sum += c->dropped.load(std::memory_order_relaxed);
            }
        }
        return sum;
    };
    auto sum_records = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->records.load(std::memory_order_relaxed);
        }
        return sum;
    };

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(cq, workload, true);  // compile + warm up

    const uint32_t trace_programs = kNumProgramsInTrace;
    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < trace_programs; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    std::atomic<bool> stop{false};
    std::thread monitor([&]() {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t last_records = 0;
        uint64_t last_published = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            // Fine (~1ms) sample of the publish stream over the interval: active ms (records arrived)
            // vs idle ms (none). active_frac ~1.0 => device producing steadily; ~0.3 => bursts then idle.
            const auto* rt_fs = mesh_device->impl().get_realtime_profiler();
            uint64_t fs_active = 0;
            uint64_t fs_idle = 0;
            uint64_t fs_max_per_ms = 0;
            uint64_t fs_last = rt_fs ? rt_fs->num_published_records() : 0;
            const auto fs_start = std::chrono::steady_clock::now();
            while (!stop.load(std::memory_order_relaxed) &&
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - fs_start).count() < print_seconds) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                const uint64_t fp = rt_fs ? rt_fs->num_published_records() : 0;
                const uint64_t fd = fp - fs_last;
                fs_last = fp;
                if (fd > 0) {
                    fs_active++;
                } else {
                    fs_idle++;
                }
                if (fd > fs_max_per_ms) {
                    fs_max_per_ms = fd;
                }
            }
            const uint64_t total = sum_records();
            const uint64_t delta = total - last_records;
            last_records = total;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const double spike_elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - spike_t0).count();
            const bool spike_on = (static_cast<long long>(spike_elapsed / spike_period) % 2) == 1;
            const auto* rt = mesh_device->impl().get_realtime_profiler();
            const uint64_t pub_records = rt ? rt->num_published_records() : 0;
            const uint64_t pub_delta = pub_records - last_published;
            last_published = pub_records;
            const uint64_t pub_batches = rt ? rt->num_published_batches() : 0;
            const double mean_pub =
                pub_batches ? static_cast<double>(pub_records) / static_cast<double>(pub_batches) : 0.0;
            const uint64_t fs_total = fs_active + fs_idle;
            log_info(
                tt::LogTest,
                "[RT mixed t={:6.1f}s] device={:.0f} rec/s | recv {:.0f}/s over {} cb | max_fifo={} pages "
                "mean_pub={:.1f} | active_frac={:.2f} max_rec_per_ms={} | drop fast={} slow={} spiky(spike={})={}",
                elapsed,
                static_cast<double>(pub_delta) / print_seconds,
                static_cast<double>(delta) / print_seconds,
                num_callbacks,
                rt ? rt->peak_fifo_pages() : 0u,
                mean_pub,
                fs_total ? static_cast<double>(fs_active) / static_cast<double>(fs_total) : 0.0,
                fs_max_per_ms,
                group_dropped(Behavior::Fast),
                group_dropped(Behavior::Slow),
                spike_on,
                group_dropped(Behavior::Spiky));
        }
    });

    const auto t_start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() < run_seconds) {
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
    }

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    stop.store(true, std::memory_order_relaxed);
    monitor.join();

    log_info(
        tt::LogTest,
        "[RT mixed FINAL] total_records={} | fast: rec={} drop={} | slow: rec={} drop={} | spiky: rec={} drop={}",
        sum_records(),
        group_records(Behavior::Fast),
        group_dropped(Behavior::Fast),
        group_records(Behavior::Slow),
        group_dropped(Behavior::Slow),
        group_records(Behavior::Spiky),
        group_dropped(Behavior::Spiky));

    for (auto handle : handles) {
        UnregisterProgramRealtimeProfilerCallback(handle);
    }
    mesh_device->release_mesh_trace(trace_id);

    EXPECT_GT(sum_records(), 0u) << "No RT profiler records received during the mixed-callback stress run";
    // Fast readers keep up, so the slow/spiky groups must drop at least as much as the fast group.
    EXPECT_GE(group_dropped(Behavior::Slow), group_dropped(Behavior::Fast));
    EXPECT_GE(group_dropped(Behavior::Spiky), group_dropped(Behavior::Fast));
    EXPECT_TRUE(mesh_device->close());
}

// Bursty replay stress for callback consumers that sleep after an empty ring read.
// Uses short peak-rate trace replays with gaps near the 1ms idle backoff so
// the consumer repeatedly transitions between active draining and idle sleep.
//
//   TT_RT_PROFILER_BURSTY_STRESS_SECONDS=300 ./<binary> --gtest_filter='*BurstyTraceReplayCallbackIdleBackoff'
//   TT_RT_PROFILER_BURSTY_STRESS_PRINT_SEC=5  (optional; default 10s between prints)
//   TT_RT_PROFILER_BURSTY_STRESS_CALLBACKS=8  (optional; default 1)
//   TT_RT_PROFILER_BURSTY_STRESS_PROGRAMS_PER_BURST=32
//   TT_RT_PROFILER_BURSTY_STRESS_GAP_PROFILE=spin
//   TT_RT_PROFILER_BURSTY_STRESS_EXPECT_CALLBACK_DROPS=1
TEST(RealtimeProfilerStress, BurstyTraceReplayCallbackIdleBackoff) {
    const char* secs_env = std::getenv(BurstyStressConfig::kSecondsEnv);
    if (secs_env == nullptr) {
        GTEST_SKIP() << "Set " << BurstyStressConfig::kSecondsEnv << "=<n> to run the bursty callback stress";
    }
    const double run_seconds = std::atof(secs_env);
    if (run_seconds <= 0.0) {
        GTEST_SKIP() << BurstyStressConfig::kSecondsEnv << " must be positive";
    }
    const char* print_env = std::getenv(BurstyStressConfig::kPrintSecondsEnv);
    const double print_seconds =
        (print_env != nullptr) ? std::max(0.1, std::atof(print_env)) : BurstyStressConfig::kDefaultPrintSeconds;
    const char* callbacks_env = std::getenv(BurstyStressConfig::kCallbacksEnv);
    const uint32_t num_callbacks =
        (callbacks_env != nullptr) ? static_cast<uint32_t>(std::max(1, std::atoi(callbacks_env))) : 1;
    const bool expect_callback_drops = env_flag_enabled(std::getenv(BurstyStressConfig::kExpectCallbackDropsEnv));
    const char* programs_per_burst_env = std::getenv(BurstyStressConfig::kProgramsPerBurstEnv);
    const uint32_t programs_per_burst = (programs_per_burst_env != nullptr)
                                            ? static_cast<uint32_t>(std::max(1, std::atoi(programs_per_burst_env)))
                                            : BurstyStressConfig::kProgramsPerBurst;
    const char* gap_profile_env = std::getenv(BurstyStressConfig::kGapProfileEnv);
    const bool use_spin_gap_profile = gap_profile_env != nullptr && std::string_view(gap_profile_env) == "spin";

    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        kTraceRegionSize,
        1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    const auto threads_before_callbacks = thread_cpu_seconds_by_tid();

    struct CallbackCounters {
        std::atomic<uint64_t> records{0};
        std::atomic<uint64_t> dropped{0};
        std::atomic<uint64_t> batches{0};
        std::atomic<uint64_t> max_batch_records{0};
    };
    std::vector<std::unique_ptr<CallbackCounters>> counters;
    std::vector<ProgramRealtimeProfilerCallbackHandle> handles;
    counters.reserve(num_callbacks);
    handles.reserve(num_callbacks);
    for (uint32_t i = 0; i < num_callbacks; ++i) {
        counters.push_back(std::make_unique<CallbackCounters>());
        CallbackCounters* c = counters.back().get();
        handles.push_back(RegisterProgramRealtimeProfilerCallback([c](const ProgramRealtimeRecordBatch& batch) {
            c->records.fetch_add(batch.records.size(), std::memory_order_relaxed);
            c->dropped.fetch_add(batch.dropped, std::memory_order_relaxed);
            c->batches.fetch_add(1, std::memory_order_relaxed);
            update_max(c->max_batch_records, batch.records.size());
        }));
    }
    const auto threads_after_callbacks = thread_cpu_seconds_by_tid();

    auto sum_records = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->records.load(std::memory_order_relaxed);
        }
        return sum;
    };
    auto sum_dropped = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->dropped.load(std::memory_order_relaxed);
        }
        return sum;
    };
    auto sum_batches = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->batches.load(std::memory_order_relaxed);
        }
        return sum;
    };
    auto max_callback_batch = [&counters]() {
        uint64_t max_value = 0;
        for (const auto& c : counters) {
            max_value = std::max(max_value, c->max_batch_records.load(std::memory_order_relaxed));
        }
        return max_value;
    };

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(cq, workload, true);

    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < programs_per_burst; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> replays{0};
    std::atomic<uint64_t> requested_programs{0};
    std::atomic<uint64_t> gaps_at_or_above_idle{0};
    std::atomic<uint64_t> max_trace_len{0};
    std::atomic<uint64_t> max_gap_us{0};
    std::thread monitor([&]() {
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t last_records = 0;
        while (!stop.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::duration<double>(print_seconds));
            const uint64_t total = sum_records();
            const uint64_t delta = total - last_records;
            last_records = total;
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            const auto* rt = mesh_device->impl().get_realtime_profiler();
            const uint64_t pub_records = rt ? rt->num_published_records() : 0;
            const uint64_t pub_batches = rt ? rt->num_published_batches() : 0;
            const double mean_publish_batch = pub_batches ? static_cast<double>(pub_records) / pub_batches : 0.0;
            log_info(
                tt::LogTest,
                "[RT bursty t={:6.1f}s] records +{} ({:.0f}/s over {} callbacks) total={} batches={} "
                "max_callback_batch={} callback_dropped={} replays={} requested_programs={} max_trace={} "
                "max_gap_us={} gaps_ge_1ms={} gap_profile={} programs_per_burst={} | receiver: max_fifo={} pages, "
                "mean_publish_batch={:.1f}",
                elapsed,
                delta,
                static_cast<double>(delta) / print_seconds,
                num_callbacks,
                total,
                sum_batches(),
                max_callback_batch(),
                sum_dropped(),
                replays.load(std::memory_order_relaxed),
                requested_programs.load(std::memory_order_relaxed),
                max_trace_len.load(std::memory_order_relaxed),
                max_gap_us.load(std::memory_order_relaxed),
                gaps_at_or_above_idle.load(std::memory_order_relaxed),
                use_spin_gap_profile ? "spin" : "idle",
                programs_per_burst,
                rt ? rt->peak_fifo_pages() : 0u,
                mean_publish_batch);
        }
    });

    uint64_t rng = 0x9e3779b97f4a7c15ull;
    auto next_random = [&rng]() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        return rng;
    };

    const auto thread_cpu_start = thread_cpu_seconds_by_tid();
    const auto t_start = std::chrono::steady_clock::now();
    const double cpu_start = process_cpu_seconds();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() < run_seconds) {
        const uint64_t r = next_random();
        mesh_device->replay_mesh_trace(cq.id(), trace_id, true);
        replays.fetch_add(1, std::memory_order_relaxed);
        requested_programs.fetch_add(programs_per_burst, std::memory_order_relaxed);
        update_max(max_trace_len, programs_per_burst);

        const uint32_t gap_us =
            use_spin_gap_profile
                ? BurstyStressConfig::kSpinGapMicros[(r >> 8) % BurstyStressConfig::kSpinGapMicros.size()]
                : BurstyStressConfig::kGapMicros[(r >> 8) % BurstyStressConfig::kGapMicros.size()];
        update_max(max_gap_us, gap_us);
        if (gap_us >= BurstyStressConfig::kIdleBackoffMicros) {
            gaps_at_or_above_idle.fetch_add(1, std::memory_order_relaxed);
        }
        if (gap_us != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(gap_us));
        }
    }
    const auto t_end = std::chrono::steady_clock::now();
    const double cpu_end = process_cpu_seconds();
    const auto thread_cpu_end = thread_cpu_seconds_by_tid();
    const double measured_wall_seconds = std::chrono::duration<double>(t_end - t_start).count();
    const double cpu_percent =
        measured_wall_seconds > 0.0 ? 100.0 * (cpu_end - cpu_start) / measured_wall_seconds : 0.0;
    double callback_thread_cpu_seconds = 0.0;
    uint32_t callback_thread_count = 0;
    for (const auto& [tid, unused] : threads_after_callbacks) {
        if (threads_before_callbacks.contains(tid)) {
            continue;
        }
        const auto start_it = thread_cpu_start.find(tid);
        const auto end_it = thread_cpu_end.find(tid);
        if (start_it == thread_cpu_start.end() || end_it == thread_cpu_end.end()) {
            continue;
        }
        callback_thread_cpu_seconds += std::max(0.0, end_it->second - start_it->second);
        callback_thread_count++;
    }
    const double callback_thread_cpu_percent =
        measured_wall_seconds > 0.0 ? 100.0 * callback_thread_cpu_seconds / measured_wall_seconds : 0.0;

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    stop.store(true, std::memory_order_relaxed);
    monitor.join();

    const auto* final_rt = mesh_device->impl().get_realtime_profiler();
    const uint64_t final_pub_records = final_rt ? final_rt->num_published_records() : 0;
    const uint64_t final_pub_batches = final_rt ? final_rt->num_published_batches() : 0;
    const double final_mean_publish_batch =
        final_pub_batches ? static_cast<double>(final_pub_records) / final_pub_batches : 0.0;
    log_info(
        tt::LogTest,
        "[RT bursty FINAL] callbacks={} total_records={} batches={} max_callback_batch={} callback_dropped={} "
        "replays={} requested_programs={} max_trace={} max_gap_us={} gaps_ge_1ms={} | receiver: "
        "max_fifo={} pages, mean_publish_batch={:.1f}, gap_profile={}, programs_per_burst={}, "
        "process_cpu_percent={:.1f}, callback_threads={}, callback_thread_cpu_percent={:.1f}",
        num_callbacks,
        sum_records(),
        sum_batches(),
        max_callback_batch(),
        sum_dropped(),
        replays.load(std::memory_order_relaxed),
        requested_programs.load(std::memory_order_relaxed),
        max_trace_len.load(std::memory_order_relaxed),
        max_gap_us.load(std::memory_order_relaxed),
        gaps_at_or_above_idle.load(std::memory_order_relaxed),
        final_rt ? final_rt->peak_fifo_pages() : 0u,
        final_mean_publish_batch,
        use_spin_gap_profile ? "spin" : "idle",
        programs_per_burst,
        cpu_percent,
        callback_thread_count,
        callback_thread_cpu_percent);

    for (auto handle : handles) {
        UnregisterProgramRealtimeProfilerCallback(handle);
    }
    mesh_device->release_mesh_trace(trace_id);

    EXPECT_GT(replays.load(std::memory_order_relaxed), 0u);
    EXPECT_GT(gaps_at_or_above_idle.load(std::memory_order_relaxed), 0u)
        << "The bursty stress did not exercise gaps at or above the consumer idle backoff";
    EXPECT_GT(sum_records(), 0u) << "No RT profiler records received during the bursty stress run";
    EXPECT_GT(sum_batches(), 0u) << "No callback batches delivered during the bursty stress run";
    EXPECT_GT(max_callback_batch(), 1u) << "The callback consumer never accumulated a burst";
    if (expect_callback_drops) {
        EXPECT_GT(sum_dropped(), 0u) << "Expected callback drops from bursty idle-backoff stress";
    } else {
        EXPECT_EQ(sum_dropped(), 0u) << "Fast callbacks dropped records under bursty replay pressure";
    }
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerStress, CallbackIdleCpuUsage) {
    const char* secs_env = std::getenv(IdleCpuConfig::kSecondsEnv);
    if (secs_env == nullptr) {
        GTEST_SKIP() << "Set " << IdleCpuConfig::kSecondsEnv << "=<n> to run the callback idle CPU test";
    }
    const double run_seconds = std::atof(secs_env);
    if (run_seconds <= 0.0) {
        GTEST_SKIP() << IdleCpuConfig::kSecondsEnv << " must be positive";
    }
    const char* callbacks_env = std::getenv(IdleCpuConfig::kCallbacksEnv);
    const uint32_t num_callbacks =
        (callbacks_env != nullptr) ? static_cast<uint32_t>(std::max(1, std::atoi(callbacks_env))) : 1;
    const char* warmup_env = std::getenv(IdleCpuConfig::kWarmupMillisEnv);
    const auto warmup = warmup_env != nullptr ? std::chrono::milliseconds(std::max(0, std::atoi(warmup_env)))
                                              : IdleCpuConfig::kDefaultWarmup;

    auto mesh_device = distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(std::nullopt),
        DEFAULT_L1_SMALL_SIZE,
        kTraceRegionSize,
        1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    const auto threads_before_callbacks = thread_cpu_seconds_by_tid();

    struct CallbackCounters {
        std::atomic<uint64_t> records{0};
        std::atomic<uint64_t> dropped{0};
    };
    std::vector<std::unique_ptr<CallbackCounters>> counters;
    std::vector<ProgramRealtimeProfilerCallbackHandle> handles;
    counters.reserve(num_callbacks);
    handles.reserve(num_callbacks);
    for (uint32_t i = 0; i < num_callbacks; ++i) {
        counters.push_back(std::make_unique<CallbackCounters>());
        CallbackCounters* c = counters.back().get();
        handles.push_back(RegisterProgramRealtimeProfilerCallback([c](const ProgramRealtimeRecordBatch& batch) {
            c->records.fetch_add(batch.records.size(), std::memory_order_relaxed);
            c->dropped.fetch_add(batch.dropped, std::memory_order_relaxed);
        }));
    }

    auto sum_records = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->records.load(std::memory_order_relaxed);
        }
        return sum;
    };
    auto sum_dropped = [&counters]() {
        uint64_t sum = 0;
        for (const auto& c : counters) {
            sum += c->dropped.load(std::memory_order_relaxed);
        }
        return sum;
    };

    std::this_thread::sleep_for(warmup);
    const auto thread_cpu_start = thread_cpu_seconds_by_tid();
    const auto t_start = std::chrono::steady_clock::now();
    const double cpu_start = process_cpu_seconds();
    std::this_thread::sleep_for(std::chrono::duration<double>(run_seconds));
    const double cpu_end = process_cpu_seconds();
    const auto t_end = std::chrono::steady_clock::now();
    const auto thread_cpu_end = thread_cpu_seconds_by_tid();

    const double measured_wall_seconds = std::chrono::duration<double>(t_end - t_start).count();
    const double process_cpu_percent =
        measured_wall_seconds > 0.0 ? 100.0 * (cpu_end - cpu_start) / measured_wall_seconds : 0.0;
    double callback_thread_cpu_seconds = 0.0;
    uint32_t callback_thread_count = 0;
    for (const auto& [tid, start_seconds] : thread_cpu_start) {
        if (threads_before_callbacks.contains(tid)) {
            continue;
        }
        const auto end_it = thread_cpu_end.find(tid);
        if (end_it == thread_cpu_end.end()) {
            continue;
        }
        callback_thread_cpu_seconds += std::max(0.0, end_it->second - start_seconds);
        callback_thread_count++;
    }
    const double callback_thread_cpu_percent =
        measured_wall_seconds > 0.0 ? 100.0 * callback_thread_cpu_seconds / measured_wall_seconds : 0.0;

    // Isolate the receiver thread (named "RealtimeProfiler", truncated to 15 chars by Linux comm) from the
    // process total, which also includes the tt-metal runtime's own idle polling threads.
    const auto comms = thread_comm_by_tid();
    double receiver_thread_cpu_seconds = 0.0;
    for (const auto& [tid, start_seconds] : thread_cpu_start) {
        const auto comm_it = comms.find(tid);
        if (comm_it == comms.end() || comm_it->second != "RealtimeProfile") {
            continue;
        }
        const auto end_it = thread_cpu_end.find(tid);
        if (end_it == thread_cpu_end.end()) {
            continue;
        }
        receiver_thread_cpu_seconds += std::max(0.0, end_it->second - start_seconds);
    }
    const double receiver_thread_cpu_percent =
        measured_wall_seconds > 0.0 ? 100.0 * receiver_thread_cpu_seconds / measured_wall_seconds : 0.0;

    log_info(
        tt::LogTest,
        "[RT idle CPU FINAL] callbacks={} callback_threads={} wall_seconds={:.3f} process_cpu_seconds={:.6f} "
        "process_cpu_percent={:.3f} receiver_thread_cpu_percent={:.3f} callback_thread_cpu_seconds={:.6f} "
        "callback_thread_cpu_percent={:.3f} records={} callback_dropped={}",
        num_callbacks,
        callback_thread_count,
        measured_wall_seconds,
        cpu_end - cpu_start,
        process_cpu_percent,
        receiver_thread_cpu_percent,
        callback_thread_cpu_seconds,
        callback_thread_cpu_percent,
        sum_records(),
        sum_dropped());

    for (auto handle : handles) {
        UnregisterProgramRealtimeProfilerCallback(handle);
    }

    EXPECT_EQ(sum_dropped(), 0u);
    EXPECT_GE(callback_thread_count, num_callbacks);
    EXPECT_TRUE(mesh_device->close());
}

namespace ReactivityConfig {
constexpr const char* kEnableEnv = "TT_RT_PROFILER_REACTIVITY";
constexpr const char* kOpsEnv = "TT_RT_PROFILER_REACTIVITY_OPS";
constexpr uint32_t kDefaultOps = 200;
// Host-paced op-to-op gaps swept (microseconds): the consumer spin window (~a few us), the receiver idle
// poll backoff (~100us), and well beyond it. The back-to-back floor is measured separately via a burst
// trace -- pacing can't reach it (the host issues faster than the device drains).
inline constexpr std::array<uint32_t, 8> kGapMicros = {5, 20, 50, 100, 200, 500, 1000, 5000};
}  // namespace ReactivityConfig

// Measures end-to-end delivery latency (op dispatch -> callback delivers that op's record) across a sweep
// of op-to-op gaps, to check the consumer wakes promptly instead of oversleeping. Ops are issued as
// non-blocking replays of a 1-op trace, paced with a busy-wait, so the small gaps are actually hit -- a
// per-op EnqueueMeshWorkload's host overhead would floor them at ~tens of us. Trace ops all carry one
// runtime_id, and records are FIFO, so the k-th matching record is the k-th replay: match by order. The
// latency beyond the gap is the host-pipeline reactiveness (receiver poll + consumer wake); pair with the
// !!RealtimeProfilerConsumerWait Tracy zone for the per-wait breakdown.
//
// The back-to-back floor (op-to-op ~= the device's record rate) is measured separately: one replay of a
// burst trace holding ops_per_gap copies, so the device fires them with no host pacing, and the rate is
// read from the delivery cadence rather than the host issue rate.
//
//   TT_RT_PROFILER_REACTIVITY=1 ./<binary> --gtest_filter='*CallbackReactivity'
//   TT_RT_PROFILER_REACTIVITY_OPS=200  (optional; ops per gap bucket)
TEST(RealtimeProfilerStress, CallbackReactivity) {
    if (!env_flag_enabled(std::getenv(ReactivityConfig::kEnableEnv))) {
        GTEST_SKIP() << "Set " << ReactivityConfig::kEnableEnv << "=1 to run the callback reactivity sweep";
    }
    const char* ops_env = std::getenv(ReactivityConfig::kOpsEnv);
    const uint32_t ops_per_gap =
        (ops_env != nullptr) ? static_cast<uint32_t>(std::max(1, std::atoi(ops_env))) : ReactivityConfig::kDefaultOps;
    const auto& gaps = ReactivityConfig::kGapMicros;
    const uint32_t num_gaps = static_cast<uint32_t>(gaps.size());
    const uint32_t total_ops = ops_per_gap * num_gaps;

    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, kTraceRegionSize, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    // Replayed trace ops carry this id; warmup/flush dispatches use kStressRuntimeId so their records are
    // ignored. Records are FIFO with no drops, so the k-th kReactivityId record is the k-th replay --
    // enqueued[k] and delivered[k] line up by index. delivered[] is filled in delivery order by the
    // callback thread; enqueued[] in issue order by this thread.
    constexpr uint32_t kReactivityId = 0x6AC0;
    constexpr uint32_t kMaxRateId = 0x6AC1;  // burst-trace ops, for the back-to-back floor
    std::vector<std::chrono::steady_clock::time_point> enqueued(total_ops);
    std::vector<std::atomic<std::chrono::steady_clock::rep>> delivered(total_ops);  // 0 == not yet delivered
    std::vector<std::atomic<uint64_t>> burst_start(ops_per_gap);                    // device start ticks, order-matched
    std::vector<std::atomic<std::chrono::steady_clock::rep>> burst_delivered(ops_per_gap);  // host delivery time
    std::atomic<uint64_t> deliver_idx{0};
    std::atomic<uint64_t> burst_idx{0};
    std::atomic<double> burst_freq{0.0};  // device cycles per ns
    std::atomic<uint64_t> dropped_total{0};

    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
            dropped_total.fetch_add(batch.dropped, std::memory_order_relaxed);
            const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
            for (const auto& rec : batch.records) {
                if (rec.runtime_id == kReactivityId) {
                    const uint64_t idx = deliver_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx < total_ops) {
                        delivered[idx].store(now, std::memory_order_relaxed);
                    }
                } else if (rec.runtime_id == kMaxRateId) {
                    const uint64_t idx = burst_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx < ops_per_gap) {
                        burst_start[idx].store(rec.start_timestamp, std::memory_order_relaxed);
                        burst_delivered[idx].store(now, std::memory_order_relaxed);
                        burst_freq.store(rec.frequency, std::memory_order_relaxed);
                    }
                }
            }
        });

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    auto set_id = [&workload](uint32_t id) {
        for (auto& [_, prog] : workload.get_programs()) {
            prog.set_runtime_id(static_cast<uint64_t>(id));
        }
    };
    auto spin_until = [](std::chrono::steady_clock::time_point deadline) {
        while (std::chrono::steady_clock::now() < deadline) {
        }
    };

    distributed::EnqueueMeshWorkload(cq, workload, true);  // compile + warm up (id != kReactivityId, ignored)

    // 1-op trace (kReactivityId) for the paced sweep; a burst trace of ops_per_gap copies (kMaxRateId) for
    // the back-to-back floor. Replaying these from a pre-assembled stream drops the per-op host cost to ~us.
    set_id(kReactivityId);
    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    set_id(kMaxRateId);
    distributed::MeshTraceId burst_trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    for (uint32_t i = 0; i < ops_per_gap; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, false);
    }
    mesh_device->end_mesh_trace(cq.id(), burst_trace_id);

    // Back-to-back floor: the smallest op-to-op in the sweep. One replay fires every burst op with no host
    // pacing, so they come out at the device's record rate -- faster than host-paced gaps can reach. Run it
    // first so the sweep proceeds in increasing op-to-op order; the burst's tail flushes on the first paced op.
    mesh_device->quiesce_devices();
    mesh_device->replay_mesh_trace(cq.id(), burst_trace_id, true);

    uint32_t k = 0;
    for (uint32_t gi = 0; gi < num_gaps; ++gi) {
        // Drain the previous bucket so its device-side backlog doesn't inflate this bucket's latency.
        mesh_device->quiesce_devices();
        const auto gap = std::chrono::microseconds(gaps[gi]);
        const auto bucket_start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < ops_per_gap; ++i) {
            spin_until(bucket_start + gap * i);
            mesh_device->replay_mesh_trace(cq.id(), trace_id, false);  // non-blocking: pacing comes from spin_until
            enqueued[k] = std::chrono::steady_clock::now();
            ++k;
        }
    }

    // Each op's record is only emitted on the next dispatch (device-side flush), so these trailing dispatches
    // flush the last paced op's record. They carry kStressRuntimeId, so the matcher ignores them.
    set_id(kStressRuntimeId);
    for (uint32_t s = 0; s < 16; ++s) {
        distributed::EnqueueMeshWorkload(cq, workload, true);
    }
    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(trace_id);
    mesh_device->release_mesh_trace(burst_trace_id);

    auto percentile = [](std::vector<double>& v, double p) {
        if (v.empty()) {
            return 0.0;
        }
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1) + 0.5))];
    };

    // Back-to-back floor (the sweep's smallest op-to-op): op-to-op from the records' device timestamps (not
    // host delivery, which arrives in batches), so it reflects the rate the device actually emitted records --
    // the spacing the Tracy Program zones show. frequency is device cycles per ns.
    const uint64_t burst_matched = std::min<uint64_t>(burst_idx.load(), ops_per_gap);
    double back_to_back_us = 0.0;
    std::vector<double> burst_lag_us;
    const double freq = burst_freq.load(std::memory_order_relaxed);
    if (burst_matched >= 8 && freq > 0.0) {
        const uint64_t first = burst_start[0].load(std::memory_order_relaxed);
        const uint64_t last = burst_start[burst_matched - 1].load(std::memory_order_relaxed);
        back_to_back_us = static_cast<double>(last - first) / static_cast<double>(burst_matched - 1) / freq / 1000.0;
        // Trim transients: skip the first quarter (the burst starts into an idle receiver, so those carry its
        // wake-from-idle latency) and the last op (its record only flushes on the next dispatch, so it is
        // delivered late). The rest is steady-state -- how far each delivery drifts behind the device's
        // production cadence; ~0 means the consumer is keeping up at the max rate.
        const uint64_t warmup = burst_matched / 4;
        const auto d0 = burst_delivered[warmup].load(std::memory_order_relaxed);
        const uint64_t s0 = burst_start[warmup].load(std::memory_order_relaxed);
        burst_lag_us.reserve(burst_matched - warmup - 2);
        for (uint64_t k = warmup + 1; k + 1 < burst_matched; ++k) {
            const double delivered_rel_us = (burst_delivered[k].load(std::memory_order_relaxed) - d0) / 1000.0;
            const double produced_rel_us =
                static_cast<double>(burst_start[k].load(std::memory_order_relaxed) - s0) / freq / 1000.0;
            burst_lag_us.push_back(std::max(0.0, delivered_rel_us - produced_rel_us));
        }
    }
    log_info(
        tt::LogTest,
        "[RT reactivity] back-to-back (burst trace) op-to-op={:.2f}us delivered={}/{} | consumer lag p50={:.1f} "
        "p99={:.1f} max={:.1f}us",
        back_to_back_us,
        burst_matched,
        ops_per_gap,
        percentile(burst_lag_us, 0.50),
        percentile(burst_lag_us, 0.99),
        percentile(burst_lag_us, 1.0));

    const uint64_t matched = std::min<uint64_t>(deliver_idx.load(), total_ops);
    double worst_overhead_p99_us = 0.0;
    for (uint32_t gi = 0; gi < num_gaps; ++gi) {
        std::vector<double> latency_us;
        std::vector<double> overhead_us;
        latency_us.reserve(ops_per_gap);
        for (uint32_t i = 0; i < ops_per_gap; ++i) {
            const uint32_t idx = gi * ops_per_gap + i;
            const auto d = delivered[idx].load(std::memory_order_relaxed);
            if (d == 0) {
                continue;
            }
            const auto delivered_tp = std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(d));
            const double lat = std::chrono::duration<double, std::micro>(delivered_tp - enqueued[idx]).count();
            latency_us.push_back(lat);
            overhead_us.push_back(std::max(0.0, lat - static_cast<double>(gaps[gi])));
        }
        const double ov_p99 = percentile(overhead_us, 0.99);
        worst_overhead_p99_us = std::max(worst_overhead_p99_us, ov_p99);
        double achieved_us = 0.0;
        if (ops_per_gap > 1) {
            achieved_us = std::chrono::duration<double, std::micro>(
                              enqueued[gi * ops_per_gap + ops_per_gap - 1] - enqueued[gi * ops_per_gap])
                              .count() /
                          (ops_per_gap - 1);
        }
        log_info(
            tt::LogTest,
            "[RT reactivity] gap={:6}us achieved={:7.1f}us delivered={}/{} | latency p50={:.1f} p99={:.1f} "
            "max={:.1f}us | overhead p50={:.1f} p99={:.1f}us",
            gaps[gi],
            achieved_us,
            latency_us.size(),
            ops_per_gap,
            percentile(latency_us, 0.50),
            percentile(latency_us, 0.99),
            percentile(latency_us, 1.0),
            percentile(overhead_us, 0.50),
            ov_p99);
    }

    // Order-matching is only valid with no drops; the trivial callback should never drop here.
    EXPECT_EQ(dropped_total.load(), 0u) << "callback dropped records; the order-matched latencies are unreliable";
    EXPECT_GE(burst_matched, ops_per_gap - 1)
        << "only " << burst_matched << "/" << ops_per_gap << " back-to-back burst ops reached the callback";
    EXPECT_GE(matched, total_ops - total_ops / 100)
        << "only " << matched << "/" << total_ops << " ops were delivered to the callback";
    EXPECT_LT(worst_overhead_p99_us, 50000.0)
        << "delivery overhead beyond the op-to-op gap reached " << worst_overhead_p99_us
        << "us (worst p99); the consumer is not waking promptly";
    EXPECT_TRUE(mesh_device->close());
}

// Drives a sustained fast op-to-op stream (paced trace replays) and measures the receiver thread's CPU in
// isolation, to compare the spin window's busy-poll cost against a sleep-only backoff. Tune the receiver via
// TT_RT_PROFILER_SPIN_US / TT_RT_PROFILER_BACKOFF_FLOOR_US.
//
//   TT_RT_PROFILER_RECEIVER_CPU=1 [TT_RT_PROFILER_RECEIVER_CPU_OP_US=2] [TT_RT_PROFILER_RECEIVER_CPU_SECONDS=2] \
//     TT_RT_PROFILER_SPIN_US=4 TT_RT_PROFILER_BACKOFF_FLOOR_US=4 ./<binary> --gtest_filter='*ReceiverCpuUnderLoad'
TEST(RealtimeProfilerStress, ReceiverCpuUnderLoad) {
    if (!env_flag_enabled(std::getenv("TT_RT_PROFILER_RECEIVER_CPU"))) {
        GTEST_SKIP() << "Set TT_RT_PROFILER_RECEIVER_CPU=1 to run the receiver CPU load test";
    }
    auto env_u32 = [](const char* name, uint32_t fallback) {
        const char* v = std::getenv(name);
        return v != nullptr ? static_cast<uint32_t>(std::max(1, std::atoi(v))) : fallback;
    };
    const uint32_t op_us = env_u32("TT_RT_PROFILER_RECEIVER_CPU_OP_US", 2);
    const uint32_t seconds = env_u32("TT_RT_PROFILER_RECEIVER_CPU_SECONDS", 2);

    constexpr int kDeviceId = 0;
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId, DEFAULT_L1_SMALL_SIZE, kTraceRegionSize, 1, DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::atomic<uint64_t> delivered{0};
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&delivered](const ProgramRealtimeRecordBatch& batch) {
            delivered.fetch_add(batch.records.size(), std::memory_order_relaxed);
        });

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(cq, workload, true);  // compile + warm up
    distributed::MeshTraceId trace_id = distributed::BeginTraceCapture(mesh_device.get(), cq.id());
    distributed::EnqueueMeshWorkload(cq, workload, false);
    mesh_device->end_mesh_trace(cq.id(), trace_id);

    auto spin_until = [](std::chrono::steady_clock::time_point deadline) {
        while (std::chrono::steady_clock::now() < deadline) {
        }
    };

    const auto comms = thread_comm_by_tid();
    const auto cpu_start = thread_cpu_seconds_by_tid();
    const auto wall_start = std::chrono::steady_clock::now();
    const auto deadline = wall_start + std::chrono::seconds(seconds);
    const auto op = std::chrono::microseconds(op_us);
    uint64_t issued = 0;
    auto next = wall_start;
    while (std::chrono::steady_clock::now() < deadline) {
        spin_until(next);
        mesh_device->replay_mesh_trace(cq.id(), trace_id, false);  // non-blocking: pacing comes from spin_until
        ++issued;
        next += op;
    }
    const auto wall_end = std::chrono::steady_clock::now();
    const auto cpu_end = thread_cpu_seconds_by_tid();

    distributed::EnqueueMeshWorkload(cq, workload, true);  // flush the tail
    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    UnregisterProgramRealtimeProfilerCallback(handle);
    mesh_device->release_mesh_trace(trace_id);

    const double wall_s = std::chrono::duration<double>(wall_end - wall_start).count();
    double receiver_cpu_s = 0.0;
    for (const auto& [tid, start_s] : cpu_start) {
        const auto comm_it = comms.find(tid);
        if (comm_it == comms.end() || comm_it->second != "RealtimeProfile") {
            continue;
        }
        const auto end_it = cpu_end.find(tid);
        if (end_it == cpu_end.end()) {
            continue;
        }
        receiver_cpu_s += std::max(0.0, end_it->second - start_s);
    }
    const double receiver_cpu_percent = wall_s > 0.0 ? 100.0 * receiver_cpu_s / wall_s : 0.0;
    const double achieved_op_us = issued > 0 ? wall_s * 1e6 / static_cast<double>(issued) : 0.0;

    log_info(
        tt::LogTest,
        "[RT receiver CPU] target_op={}us achieved_op={:.2f}us issued={} delivered={} receiver_cpu_percent={:.1f}",
        op_us,
        achieved_op_us,
        issued,
        delivered.load(),
        receiver_cpu_percent);

    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
