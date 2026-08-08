// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sync-accuracy coverage for the real-time (RT) profiler: the published sync-error distribution, and the mapping's
// residual against an independent read of the device clock.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
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
    EXPECT_LT(it->clock_sync.error, std::chrono::nanoseconds(1500)) << "sync error is too high";
    log_info(
        tt::LogTest,
        "[RT profiler sync] long program: duration={:.3f}s sync error={}us",
        std::chrono::duration<double>{it->duration()}.count(),
        std::chrono::duration<double, std::micro>{it->clock_sync.error}.count());
    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
