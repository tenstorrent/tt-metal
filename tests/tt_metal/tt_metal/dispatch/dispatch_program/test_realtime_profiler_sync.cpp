// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sync-accuracy coverage for the real-time (RT) profiler: the published sync-error distribution, the mapping's
// residual against an independent read of the device clock, and host-only synthetic coverage of ClockSyncMapping's
// error bound under adversarial clock-rate transitions.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
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
#include <tt-metalium/tt_backend_api_types.hpp>

#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "impl/realtime_profiler/device_clock_sync.hpp"

namespace tt::tt_metal {
namespace {

using namespace std::chrono_literals;
using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeRecord;
using tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

std::string make_sync_kernel_source(uint32_t runtime_id) {
    return "#include <cstdint>\n"
           "// rt_profiler_sync_marker_" +
           std::to_string(runtime_id) +
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

Program make_sync_program(const std::string& kernel_src, const CoreRange& cores, uint32_t runtime_id) {
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

void enqueue_sync_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t runtime_id, const CoreRange& cores) {
    distributed::MeshWorkload workload;
    workload.add_program(
        distributed::MeshCoordinateRange(mesh_device->shape()),
        make_sync_program(make_sync_kernel_source(runtime_id), cores, runtime_id));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
}

std::shared_ptr<distributed::MeshDevice> open_profiler_unit_mesh() {
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        /*device_id=*/0,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
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

template <typename Predicate>
void quiesce_and_wait_for(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Predicate delivered) {
    mesh_device->quiesce_devices();
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!delivered() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
}

constexpr auto kMaxSyncError = std::chrono::microseconds(15);
constexpr auto kMaxMeanSyncError = std::chrono::microseconds(5);

TEST(RealtimeProfilerSync, SyncAccuracy) {
    auto mesh_device = open_profiler_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    std::atomic<size_t> delivered = 0;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        delivered.fetch_add(batch.records.size());
    });

    constexpr uint32_t kIterations = 300;
    for (uint32_t i = 0; i < kIterations; ++i) {
        enqueue_sync_program(mesh_device, /*runtime_id=*/1, all_cores(mesh_device));
        std::this_thread::sleep_for(kDeviceClockSyncInterval);
    }
    quiesce_and_wait_for(mesh_device, [&] { return delivered.load() >= kIterations; });
    UnregisterProgramRealtimeProfilerCallback(handle);

    ASSERT_GE(records.size(), kIterations);
    std::chrono::nanoseconds worst{};
    std::chrono::nanoseconds sum{};
    for (const auto& record : records) {
        ASSERT_GT(record.clock_sync.error, std::chrono::nanoseconds::zero()) << "sync error should be populated";
        ASSERT_LT(record.clock_sync.error, kMaxSyncError) << "maximum sync error is too high";
        worst = std::max(worst, record.clock_sync.error);
        sum += record.clock_sync.error;
    }
    const auto mean = sum / records.size();
    log_info(
        tt::LogTest,
        "[RT profiler sync] sync error over {} records: mean={}ns worst={}ns",
        records.size(),
        mean.count(),
        worst.count());
    EXPECT_LT(mean, kMaxMeanSyncError) << "average sync error is too high";
    EXPECT_TRUE(mesh_device->close());
}

// Independent check of published device_cycle_offset/frequency: stamp WALL_CLOCK into L1, bracket host reads of that
// stamp, and measure residual against the paired record's mapping.
constexpr uint32_t kClockStampIterations = 200'000'000;
constexpr uint32_t kClockCheckRounds = 3;
constexpr uint32_t kClockReadsPerRound = 100;
constexpr uint32_t kFirstClockCheckRuntimeId = 9101;
constexpr double kMaxMeanMappingErrorNs = 2'000.0;

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

TEST(RealtimeProfilerSync, RecordMappingMatchesAnIndependentClockRead) {
    auto mesh_device = open_profiler_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::vector<ProgramRealtimeRecord> records;
    std::atomic<size_t> delivered = 0;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        delivered.fetch_add(batch.records.size());
    });

    IDevice* device = mesh_device->get_devices().front();
    const uint32_t stamp_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const uint32_t stop_addr = stamp_addr + 16;
    const CoreCoord stamp_core{0, 0};
    const CoreRange grid = all_cores(mesh_device);
    auto& cluster = MetalContext::instance().get_cluster();
    const CoreCoord stamp_vcore = device->virtual_core_from_logical_core(stamp_core, CoreType::WORKER);
    const tt_cxy_pair stamp_dest(device->id(), stamp_vcore);

    const auto host_now_ns = [] { return std::chrono::steady_clock::now().time_since_epoch().count(); };
    const auto read_device_ticks = [&]() -> uint64_t {
        uint32_t words[2] = {0, 0};
        cluster.read_core(words, sizeof(words), stamp_dest, stamp_addr);
        return (static_cast<uint64_t>(words[1]) << 32) | words[0];
    };

    std::vector<ClockBracket> brackets;
    for (uint32_t round = 0; round < kClockCheckRounds; ++round) {
        const uint32_t zero = 0;
        cluster.write_core(&zero, sizeof(zero), stamp_dest, stamp_addr);
        cluster.write_core(&zero, sizeof(zero), stamp_dest, stamp_addr + 4);
        cluster.write_core_immediate(&zero, sizeof(zero), stamp_dest, stop_addr);

        Program program = CreateProgram();
        auto stamp_kernel = CreateKernelFromString(
            program,
            make_clock_stamp_kernel_source(),
            CoreRangeSet(CoreRange(stamp_core, stamp_core)),
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, stamp_kernel, stamp_core, {stamp_addr, stop_addr, kClockStampIterations});

        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);

        const auto launch_deadline = std::chrono::steady_clock::now() + 10s;
        while (read_device_ticks() == 0 && std::chrono::steady_clock::now() < launch_deadline) {
            std::this_thread::sleep_for(1ms);
        }
        ASSERT_NE(read_device_ticks(), 0u) << "the clock-stamping kernel never started";

        enqueue_sync_program(mesh_device, kFirstClockCheckRuntimeId + round, grid);

        for (uint32_t i = 0; i < kClockReadsPerRound; ++i) {
            ClockBracket bracket;
            bracket.host_before_ns = host_now_ns();
            bracket.device_ticks = read_device_ticks();
            bracket.host_after_ns = host_now_ns();
            bracket.round = round;
            brackets.push_back(bracket);
        }
        ASSERT_NE(brackets.back().device_ticks, brackets[brackets.size() - kClockReadsPerRound].device_ticks)
            << "clock stamp did not advance";

        const uint32_t stop = 1;
        cluster.write_core_immediate(&stop, sizeof(stop), stamp_dest, stop_addr);
        distributed::Finish(mesh_device->mesh_command_queue());
    }

    quiesce_and_wait_for(mesh_device, [&] { return delivered.load() >= kClockCheckRounds; });
    UnregisterProgramRealtimeProfilerCallback(handle);

    std::map<uint32_t, ProgramRealtimeRecord> record_by_runtime_id;
    for (const auto& record : records) {
        if (record.frequency > 0.0) {
            record_by_runtime_id.insert({record.runtime_id, record});
        }
    }

    double sum_residual_ns = 0.0;
    double sum_frequency = 0.0;
    size_t num_reads = 0;
    size_t reads_beyond_claim = 0;
    uint32_t rounds_checked = 0;
    for (uint32_t round = 0; round < kClockCheckRounds; ++round) {
        const auto it = record_by_runtime_id.find(kFirstClockCheckRuntimeId + round);
        if (it == record_by_runtime_id.end()) {
            continue;
        }
        ++rounds_checked;
        const ProgramRealtimeRecord& record = it->second;
        sum_frequency += record.frequency;
        log_info(
            tt::LogTest,
            "[RT profiler sync] round {}: record frequency {:.6f} GHz, claim {}ns",
            round,
            record.frequency,
            record.clock_sync.error.count());
        for (const auto& bracket : brackets) {
            if (bracket.round != round) {
                continue;
            }
            const double mapped_host_ns = (static_cast<double>(bracket.device_ticks) -
                                           static_cast<double>(record.clock_sync.device_cycle_offset)) /
                                          record.frequency;
            const double residual = mapped_host_ns - bracket.host_mid_ns();
            sum_residual_ns += residual;
            ++num_reads;
            const double excess = std::max(0.0, std::abs(residual) - bracket.half_width_ns());
            reads_beyond_claim += excess > static_cast<double>(record.clock_sync.error.count());
        }
    }
    ASSERT_GE(rounds_checked, kClockCheckRounds / 2) << "too few rounds produced a record to check against";
    ASSERT_GT(num_reads, 0u);
    const double mean_residual_ns = sum_residual_ns / static_cast<double>(num_reads);
    const double avg_frequency = sum_frequency / static_cast<double>(rounds_checked);

    log_info(
        tt::LogTest,
        "[RT profiler sync] independent clock read: {} reads over {} rounds; mean residual={:.0f}ns; avg "
        "record.frequency={:.6f} GHz; {} beyond claim",
        num_reads,
        rounds_checked,
        mean_residual_ns,
        avg_frequency,
        reads_beyond_claim);

    EXPECT_LE(std::abs(mean_residual_ns), kMaxMeanMappingErrorNs)
        << "published mapping disagrees with independent device-clock reads by more than a few microseconds";
    EXPECT_LE(reads_beyond_claim, num_reads / 100)
        << "published sync error understates the residual against independent reads too often";
    EXPECT_TRUE(mesh_device->close());
}

// The pre-streaming init sync distilled: a ~1 s dense sweep of bracketed wall-clock register reads,
// least-squares fitted. The fit is an independent estimator of the same clock, so the published
// smoothed frequency must agree with its slope to a few ppm on a quiet chip.
TEST(RealtimeProfilerSync, FrequencyMatchesAnIndependentLinearFit) {
    auto mesh_device = open_profiler_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    constexpr uint32_t kFitRuntimeIdBase = 9301;
    constexpr uint32_t kFitSamples = 600;
    constexpr uint32_t kFitPrograms = 12;
    constexpr auto kFitSampleSpacing = 1500us;

    std::vector<ProgramRealtimeRecord> records;
    std::atomic<size_t> delivered = 0;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        delivered.fetch_add(batch.records.size());
    });

    IDevice* device = mesh_device->get_devices().front();
    auto& context = MetalContext::instance();
    const uint32_t clock_addr_lo = context.hal().get_tensix_wall_clock_reg_addr_lo();
    const uint32_t clock_addr_hi = context.hal().get_tensix_wall_clock_reg_addr_hi();
    auto& cluster = context.get_cluster();
    const CoreCoord clock_vcore = device->virtual_core_from_logical_core(CoreCoord{0, 0}, CoreType::WORKER);
    const tt_cxy_pair clock_dest(device->id(), clock_vcore);
    const auto host_now_ns = [] { return std::chrono::steady_clock::now().time_since_epoch().count(); };

    struct FitSample {
        int64_t host_ns;
        uint64_t cycles;
        int64_t half_bracket_ns;
    };
    // AICLK must sit at its busy point throughout: any gap in dispatch traffic lets DVFS step the
    // clock, and each step honestly resets the frequency window, leaving nearby records with a
    // young window's wider rate. Filler programs pin it busy through the window's ~1 s maturation
    // and the sweep; only records enqueued in the final stretch are compared.
    constexpr uint32_t kFitFillerRuntimeId = 9300;
    constexpr uint32_t kComparedSpan = 300;
    uint64_t enqueued = 0;
    const auto enqueue_filler = [&] {
        enqueue_sync_program(mesh_device, kFitFillerRuntimeId, all_cores(mesh_device));
        ++enqueued;
    };
    for (int i = 0; i < 625; ++i) {
        enqueue_filler();
        std::this_thread::sleep_for(2ms);
    }
    std::vector<FitSample> samples;
    samples.reserve(kFitSamples);
    for (uint32_t i = 0; i < kFitSamples; ++i) {
        const uint32_t compared_stride = kComparedSpan / kFitPrograms;
        if (i >= kFitSamples - kComparedSpan && (i - (kFitSamples - kComparedSpan)) % compared_stride == 0) {
            enqueue_sync_program(
                mesh_device,
                kFitRuntimeIdBase + (i - (kFitSamples - kComparedSpan)) / compared_stride,
                all_cores(mesh_device));
            ++enqueued;
        } else {
            enqueue_filler();
        }
        uint32_t lo = 0;
        uint32_t lo_again = 0;
        uint32_t hi = 0;
        const int64_t before = host_now_ns();
        cluster.read_reg(&lo, clock_dest, clock_addr_lo);
        const int64_t after = host_now_ns();
        // The lo read latches hi; the second lo read discards the rare wrap that would tear the pair.
        cluster.read_reg(&hi, clock_dest, clock_addr_hi);
        cluster.read_reg(&lo_again, clock_dest, clock_addr_lo);
        if (lo_again >= lo) {
            samples.push_back(
                FitSample{(before + after) / 2, (static_cast<uint64_t>(hi) << 32) | lo, (after - before) / 2});
        }
        std::this_thread::sleep_for(kFitSampleSpacing);
    }
    quiesce_and_wait_for(mesh_device, [&] { return delivered.load() >= enqueued; });
    UnregisterProgramRealtimeProfilerCallback(handle);

    // Keep the narrowest-bracket half, then centered least squares for the slope (cycles per host ns).
    std::sort(samples.begin(), samples.end(), [](const FitSample& a, const FitSample& b) {
        return a.half_bracket_ns < b.half_bracket_ns;
    });
    samples.resize(samples.size() / 2);
    ASSERT_GT(samples.size(), 100u);
    const double x0 = static_cast<double>(samples.front().host_ns);
    const double y0 = static_cast<double>(samples.front().cycles);
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (const FitSample& sample : samples) {
        mean_x += static_cast<double>(sample.host_ns) - x0;
        mean_y += static_cast<double>(sample.cycles) - y0;
    }
    mean_x /= static_cast<double>(samples.size());
    mean_y /= static_cast<double>(samples.size());
    double sxx = 0.0;
    double sxy = 0.0;
    for (const FitSample& sample : samples) {
        const double dx = (static_cast<double>(sample.host_ns) - x0) - mean_x;
        const double dy = (static_cast<double>(sample.cycles) - y0) - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    const double fit_frequency = sxy / sxx;
    ASSERT_GT(fit_frequency, 0.0);
    double rss = 0.0;
    for (const FitSample& sample : samples) {
        const double dx = (static_cast<double>(sample.host_ns) - x0) - mean_x;
        const double dy = (static_cast<double>(sample.cycles) - y0) - mean_y;
        const double residual = dy - fit_frequency * dx;
        rss += residual * residual;
    }
    const double residual_rms_ns = std::sqrt(rss / static_cast<double>(samples.size())) / fit_frequency;
    if (residual_rms_ns > 2'000.0) {
        GTEST_SKIP() << "clock stepped mid-sweep (residual RMS " << residual_rms_ns
                     << " ns); a single linear fit is not a valid reference";
    }

    std::vector<double> ppm_deviations;
    double max_placement_delta_ns = 0.0;
    for (const auto& record : records) {
        if (record.runtime_id < kFitRuntimeIdBase || record.runtime_id >= kFitRuntimeIdBase + kFitPrograms ||
            !(record.frequency > 0.0)) {
            continue;
        }
        ppm_deviations.push_back(std::abs(record.frequency - fit_frequency) / fit_frequency * 1e6);
        const double host_by_record =
            (static_cast<double>(record.start_timestamp) - static_cast<double>(record.clock_sync.device_cycle_offset)) /
            record.frequency;
        const double host_by_fit =
            x0 + mean_x + ((static_cast<double>(record.start_timestamp) - y0) - mean_y) / fit_frequency;
        max_placement_delta_ns = std::max(max_placement_delta_ns, std::abs(host_by_record - host_by_fit));
    }
    ASSERT_EQ(ppm_deviations.size(), kFitPrograms);
    std::sort(ppm_deviations.begin(), ppm_deviations.end());
    const double median_ppm = ppm_deviations[ppm_deviations.size() / 2];
    log_info(
        tt::LogTest,
        "[RT profiler sync] independent fit: {:.9f} GHz over {} reads (residual RMS {:.0f}ns); published-vs-fit "
        "median {:.2f} ppm, max {:.2f} ppm across {} records, placement delta max {:.0f}ns",
        fit_frequency,
        samples.size(),
        residual_rms_ns,
        median_ppm,
        ppm_deviations.back(),
        ppm_deviations.size(),
        max_placement_delta_ns);
    // Median, not max: a probe-gap blip legally resets the frequency window, and records near the
    // reset ride the young window's wider rate (their reconstruction stays covered via the
    // smoothing-skew error term). Mature-window records must match the fit at the ppm scale.
    EXPECT_LT(median_ppm, 2.0) << "published frequency disagrees with an independent linear fit";
    // The placement delta also carries any constant offset between the profiler core's and this
    // worker's wall clocks, so it gets a loose sanity bound rather than a tight one.
    EXPECT_LT(max_placement_delta_ns, 50'000.0);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerSync, LongProgramIsDeliveredIntact) {
    auto mesh_device = open_profiler_unit_mesh();
    if (mesh_device == nullptr) {
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    constexpr uint32_t kLongRuntimeId = 7201;
    std::vector<ProgramRealtimeRecord> records;
    std::atomic<bool> saw_long = false;
    const auto handle = RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
        records.insert(records.end(), batch.records.begin(), batch.records.end());
        for (const auto& record : batch.records) {
            if (record.runtime_id == kLongRuntimeId) {
                saw_long.store(true);
            }
        }
    });

    const std::string long_kernel_src = R"(
void kernel_main() {
    for (unsigned i = 0; i < 62500000u; ++i) {
#pragma GCC unroll 64
        for (int j = 0; j < 64; ++j) {
            asm("nop");
        }
    }
}
)";
    Program program = CreateProgram();
    CreateKernelFromString(
        program,
        long_kernel_src,
        CoreRange(CoreCoord{0, 0}, CoreCoord{0, 0}),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    program.set_runtime_id(static_cast<uint64_t>(kLongRuntimeId));

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    const auto window_start = std::chrono::steady_clock::now();
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
    const auto window_end = std::chrono::steady_clock::now();

    quiesce_and_wait_for(mesh_device, [&] { return saw_long.load(); });
    UnregisterProgramRealtimeProfilerCallback(handle);

    const auto it = std::ranges::find_if(
        records, [](const ProgramRealtimeRecord& record) { return record.runtime_id == kLongRuntimeId; });
    ASSERT_NE(it, records.end()) << "the long-running program's record was dropped";
    EXPECT_GT(it->duration(), std::chrono::seconds(1));
    EXPECT_LT(it->duration(), std::chrono::seconds(30));
    EXPECT_GE(it->host_start(), window_start - std::chrono::milliseconds(2));
    EXPECT_GE(it->host_end(), window_start);
    EXPECT_LE(it->host_end(), window_end + std::chrono::milliseconds(2));
    EXPECT_GT(it->clock_sync.error, std::chrono::nanoseconds::zero());
    // A start predating the probe history rides back at the practical smoothed-frequency spread
    // (a few ppm on a stable clock), so the claim scales with the overhang rather than staying at
    // read noise.
    EXPECT_LT(it->clock_sync.error, std::chrono::microseconds(200)) << "sync error is too high";
    log_info(
        tt::LogTest,
        "[RT profiler sync] long program: duration={:.3f}s sync error={}us",
        std::chrono::duration<double>{it->duration()}.count(),
        std::chrono::duration<double, std::micro>{it->clock_sync.error}.count());
    EXPECT_TRUE(mesh_device->close());
}

// ---------------------------------------------------------------------------------------------------------------
// Host-only coverage of ClockSyncMapping's error bound: a synthetic device clock with adversarial rate
// transitions feeds probes to the mapping, and every mapped record must land within the published error of the
// simulated truth. Hardware cannot exercise these paths — AICLK cannot be stepped at a chosen position inside a
// chosen probe gap — so this is the only deterministic coverage of the DVFS reasoning. No device needed.

constexpr double kSpacingNs = std::chrono::duration<double, std::nano>(kDeviceClockSyncInterval).count();
constexpr double kProbeErrorNs = 500.0;

// Piecewise-constant-rate clock; the tests space transitions per the DVFS cadence except where
// they deliberately exercise the uncertified fallback.
class SyntheticDeviceClock {
public:
    explicit SyntheticDeviceClock(double rate) : segments_{{0.0, 0.0, rate}} {}

    void set_rate_at(double host_ns, double rate) { segments_.push_back({host_ns, device_at(host_ns), rate}); }

    double device_at(double host_ns) const {
        const Segment& segment = *std::prev(std::upper_bound(
            segments_.begin(), segments_.end(), host_ns, [](double t, const Segment& s) { return t < s.host_start; }));
        return segment.device_start + (host_ns - segment.host_start) * segment.rate;
    }

    double host_at(double device) const {
        const Segment& segment = *std::prev(std::upper_bound(
            segments_.begin(), segments_.end(), device, [](double d, const Segment& s) { return d < s.device_start; }));
        return segment.host_start + (device - segment.device_start) / segment.rate;
    }

private:
    struct Segment {
        double host_start;
        double device_start;
        double rate;
    };
    std::vector<Segment> segments_;
};

struct SyntheticProbeFeed {
    ClockSyncMapping& mapping;
    const SyntheticDeviceClock& clock;
    uint64_t lcg_state = 0x243F6A8885A308D3ull;  // deterministic, unstructured host noise

    // Anchors carry real host noise, uniform within the claimed read error.
    void feed(double from_host_ns, double to_host_ns, double spacing_ns = kSpacingNs) {
        for (double t = from_host_ns; t <= to_host_ns; t += spacing_ns) {
            lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
            const double unit = static_cast<double>(lcg_state >> 11) / static_cast<double>(1ull << 53);
            const double noise = (unit - 0.5) * kProbeErrorNs;
            mapping.add_probe(ClockSyncMapping::Anchor{
                .host_timestamp =
                    std::chrono::steady_clock::time_point(std::chrono::nanoseconds(std::llround(t + noise))),
                .device_timestamp = static_cast<uint64_t>(std::llround(clock.device_at(t))),
                .error = std::chrono::nanoseconds(std::llround(kProbeErrorNs)),
            });
        }
    }
};

// Maps [start_host_ns, end_host_ns] and asserts both mapped endpoints land within the published error of the
// simulated truth (plus a few ns of integer-rounding slop).
ClockSyncMapping::RecordMapping map_and_expect_covered(
    ClockSyncMapping& mapping,
    const SyntheticDeviceClock& clock,
    double start_host_ns,
    double end_host_ns,
    const std::string& what) {
    const auto start = static_cast<uint64_t>(std::llround(clock.device_at(start_host_ns)));
    const auto end = static_cast<uint64_t>(std::llround(clock.device_at(end_host_ns)));
    const auto record = mapping.map_record(start, end);
    EXPECT_TRUE(record.has_value()) << what << ": record was unmappable";
    if (!record.has_value()) {
        return {};
    }
    constexpr double kRoundingSlopNs = 4.0;
    const double claim_ns = static_cast<double>(record->error.count()) + kRoundingSlopNs;
    const auto mapped_host = [&](uint64_t device_timestamp) {
        return (static_cast<double>(device_timestamp) - static_cast<double>(record->device_cycle_offset)) /
               record->frequency;
    };
    EXPECT_LE(std::abs(mapped_host(start) - clock.host_at(static_cast<double>(start))), claim_ns)
        << what << ": start misplaced beyond the published error";
    EXPECT_LE(std::abs(mapped_host(end) - clock.host_at(static_cast<double>(end))), claim_ns)
        << what << ": end misplaced beyond the published error";
    return *record;
}

TEST(RealtimeProfilerSync, MappingQuietClockIsTightAndCovered) {
    SyntheticDeviceClock clock(1.0);
    ClockSyncMapping mapping;
    SyntheticProbeFeed feed{mapping, clock};
    feed.feed(0.0, 20.0 * kSpacingNs);

    EXPECT_GT(mapping.finalized_device_timestamp(), 0u);
    for (int k = 3; k < 18; ++k) {
        const auto record = map_and_expect_covered(
            mapping, clock, k * kSpacingNs + 60e3, k * kSpacingNs + 240e3, "quiet chord " + std::to_string(k));
        EXPECT_LT(record.error, 3us) << "quiet-clock bound should stay near read noise";
    }
}

// Head-to-head with the retired init sync (syncDeviceHost's 249-sample OLS), granted the same
// per-sample noise as our probes — strictly more charitable than its real one-way write jitter.
// Both estimate a known exact rate on a quiet clock; the window regression must not lose.
TEST(RealtimeProfilerSync, SmoothedFrequencyMatchesInitSyncRegression) {
    constexpr double kTrueRate = 0.9855;
    constexpr int kSeeds = 20;
    constexpr double kRunSpanNs = 2.5e9;
    constexpr int kInitSamples = 249;
    constexpr double kInitPeriodNs = 10e6;

    double ours_sq = 0.0;
    double init_sq = 0.0;
    for (int seed = 0; seed < kSeeds; ++seed) {
        SyntheticDeviceClock clock(kTrueRate);
        ClockSyncMapping mapping;
        SyntheticProbeFeed feed{mapping, clock};
        feed.lcg_state += static_cast<uint64_t>(seed) * 0x9E3779B97F4A7C15ull;
        feed.feed(0.0, kRunSpanNs);
        // A record inside a single, finalized chord near the end: whole-record-in-chord is the
        // path that publishes the window regression (spanning records publish their own secant).
        const double chord_open = (std::floor(kRunSpanNs / kSpacingNs) - 4.0) * kSpacingNs;
        const auto start = static_cast<uint64_t>(std::llround(clock.device_at(chord_open + 60e3)));
        const auto end = static_cast<uint64_t>(std::llround(clock.device_at(chord_open + 240e3)));
        const auto record = mapping.map_record(start, end);
        ASSERT_TRUE(record.has_value());
        ours_sq += (record->frequency - kTrueRate) * (record->frequency - kTrueRate);

        uint64_t lcg = 0x0123456789ABCDEFull + static_cast<uint64_t>(seed) * 0xD1B54A32D192ED03ull;
        double sum_t = 0.0;
        double sum_d = 0.0;
        double sum_tt = 0.0;
        double sum_td = 0.0;
        for (int i = 0; i < kInitSamples; ++i) {
            const double t = (i + 1) * kInitPeriodNs;
            lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
            const double unit = static_cast<double>(lcg >> 11) / static_cast<double>(1ull << 53);
            const double noise = (unit - 0.5) * kProbeErrorNs;
            const double d = clock.device_at(t + noise);
            sum_t += t;
            sum_d += d;
            sum_tt += t * t;
            sum_td += t * d;
        }
        const double init_freq = (kInitSamples * sum_td - sum_t * sum_d) / (kInitSamples * sum_tt - sum_t * sum_t);
        init_sq += (init_freq - kTrueRate) * (init_freq - kTrueRate);
    }

    const double ours_rms_ppm = std::sqrt(ours_sq / kSeeds) / kTrueRate * 1e6;
    const double init_rms_ppm = std::sqrt(init_sq / kSeeds) / kTrueRate * 1e6;
    log_info(
        tt::LogTest,
        "[RT profiler sync] quiet-clock frequency RMS over {} seeds: window regression={:.4f} ppm, "
        "init-sync-style OLS={:.4f} ppm",
        kSeeds,
        ours_rms_ppm,
        init_rms_ppm);
    EXPECT_LE(ours_rms_ppm, init_rms_ppm)
        << "the rolling window regression must not be less accurate than the retired init sync";
}

TEST(RealtimeProfilerSync, MappingCoversAStepLateInAChord) {
    // The failure mode of witness-style estimates: a step at 90% of a chord, where a mid-probe
    // departure test under-reports the worst-case in-chord misplacement by ~2x.
    SyntheticDeviceClock clock(0.5);
    const double step_at = 5.0 * kSpacingNs + 0.9 * kSpacingNs;
    clock.set_rate_at(step_at, 1.35);

    ClockSyncMapping mapping;
    SyntheticProbeFeed feed{mapping, clock};
    feed.feed(0.0, 12.0 * kSpacingNs);

    map_and_expect_covered(mapping, clock, 5.0 * kSpacingNs + 50e3, 5.0 * kSpacingNs + 150e3, "before the step");
    map_and_expect_covered(mapping, clock, step_at - 5e3, step_at + 5e3, "straddling the step");
    const auto at_step =
        map_and_expect_covered(mapping, clock, 5.0 * kSpacingNs + 200e3, step_at + 2e3, "ending at the step");
    EXPECT_GT(at_step.error, 5us) << "a certified chord holding a 2.7x step must own up to a large bound";
    EXPECT_LT(at_step.error, 200us);
}

TEST(RealtimeProfilerSync, MappingSurvivesTheTwoStepCancellation) {
    // Two transitions 1.15 ms apart (legal at the DVFS cadence) around one probe, with rates chosen
    // so a mid-probe departure witness measures ~zero. At 500 us probe spacing the certificate
    // windows exceed the transition spacing, so the mapping must fall back rather than trust
    // neighbors — the published error stays honest instead of collapsing to read noise.
    SyntheticDeviceClock clock(1.35);
    clock.set_rate_at(2'100e3, 0.5);
    clock.set_rate_at(3'250e3, 0.67);

    ClockSyncMapping mapping;
    SyntheticProbeFeed feed{mapping, clock};
    feed.feed(0.0, 5'000e3, /*spacing_ns=*/500e3);

    map_and_expect_covered(mapping, clock, 2'050e3, 2'150e3, "first step");
    const auto second = map_and_expect_covered(mapping, clock, 3'240e3, 3'260e3, "second step");
    EXPECT_GT(second.error, 20us) << "uncertified chord with a step must not publish a read-noise-sized error";
}

TEST(RealtimeProfilerSync, MappingCoversAReceiverStall) {
    // A probe gap far beyond the DVFS cadence, with two transitions inside it. The certificate
    // fails and the bound degrades to the fallback tier — here the chord-span containment, since
    // this feed is too short to mature a frequency window — large and honest.
    SyntheticDeviceClock clock(0.5);
    clock.set_rate_at(1'000e3, 1.35);
    clock.set_rate_at(2'000e3, 0.8);
    clock.set_rate_at(3'400e3, 1.35);
    clock.set_rate_at(4'600e3, 0.5);

    ClockSyncMapping mapping;
    SyntheticProbeFeed feed{mapping, clock};
    feed.feed(0.0, 3'000e3);
    feed.feed(5'400e3, 6'600e3);

    map_and_expect_covered(mapping, clock, 3'350e3, 3'450e3, "stall, first step");
    map_and_expect_covered(mapping, clock, 4'550e3, 4'650e3, "stall, second step");
    map_and_expect_covered(mapping, clock, 3'100e3, 5'100e3, "stall, spanning both");
}

TEST(RealtimeProfilerSync, MappingFrequencyTracksATransition) {
    SyntheticDeviceClock clock(1.0);
    const double step_at = 100.0 * kSpacingNs + 150e3;
    clock.set_rate_at(step_at, 1.2);

    ClockSyncMapping mapping;
    SyntheticProbeFeed feed{mapping, clock};
    feed.feed(0.0, 200.0 * kSpacingNs);

    const auto before =
        map_and_expect_covered(mapping, clock, 60.0 * kSpacingNs + 60e3, 60.0 * kSpacingNs + 240e3, "before step");
    EXPECT_NEAR(before.frequency, 1.0, 1.0 * 500e-6);

    // The step chord itself can only be known to its bracket; sanity, not precision.
    const auto at_step = map_and_expect_covered(mapping, clock, step_at - 40e3, step_at + 40e3, "at step");
    EXPECT_GT(at_step.frequency, 0.95);
    EXPECT_LT(at_step.frequency, 1.25);

    // A few chords later the window has restarted on the new rate and is already sub-0.5%...
    const auto soon_after = map_and_expect_covered(
        mapping, clock, 106.0 * kSpacingNs + 60e3, 106.0 * kSpacingNs + 240e3, "soon after step");
    EXPECT_NEAR(soon_after.frequency, 1.2, 1.2 * 5e-3);

    // ...and tens-of-ppm again once the window has regrown.
    const auto later =
        map_and_expect_covered(mapping, clock, 195.0 * kSpacingNs + 60e3, 195.0 * kSpacingNs + 240e3, "well after");
    EXPECT_NEAR(later.frequency, 1.2, 1.2 * 300e-6);
}

}  // namespace
}  // namespace tt::tt_metal
