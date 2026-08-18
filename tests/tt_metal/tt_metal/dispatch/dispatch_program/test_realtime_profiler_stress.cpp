// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stress test for the real-time (RT) profiler. If NCRISC or the host receiver can't drain as fast as
// BRISC produces, the device ring fills and BRISC drops records.
//
// The drain path is expected to absorb peak dispatch without backing up. Trace replay is deliberately unprofiled by
// the clean-room protocol, so stress traffic is generated with ordinary non-trace bursts.

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
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/device_command_calculator.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_commands.hpp"
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"
#include "tt_metal/distributed/realtime_profiler_manager.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

#include "realtime_profiler_test_utils.hpp"

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeRecordBatch;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

constexpr uint32_t kNumProgramsInNonTraceBurst = 4096;

constexpr uint32_t kDefaultStressSeconds = 60;

constexpr uint32_t kDefaultDropAccountingSeconds = 60;

// Ordinary launches use this runtime_id so every record we receive can be
// attributed to this test (runtime_id == 0 is intentionally unprofiled).
constexpr uint32_t kStressRuntimeId = 0xBEEFu;

// Quiesce + drain window before unregistering the callback.
constexpr auto kPostQuiesceDrain = std::chrono::milliseconds(2000);

distributed::MeshWorkload build_blank_kernel_workload(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    Program program = CreateProgram();

    // Blank kernels on BRISC + NCRISC + TRISC on a single core. Single-core
    // minimizes dispatch payload (one launch_msg per RISC, one core's worth
    // of kernel-config state), maximizing the ordinary-launch record rate.
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
        DEFAULT_TRACE_REGION_SIZE,
        1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
}

TEST(RealtimeProfilerStress, NonTraceBurstDriverPreservesRecords) {
    auto mesh_device = open_full_mesh();
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }
    const auto* rt = mesh_device->impl().get_realtime_profiler();
    ASSERT_NE(rt, nullptr);

    test_utils::RealtimeProfilerRecordCollector collector;
    ProgramRealtimeProfilerCallbackHandle handle = RegisterProgramRealtimeProfilerCallback(
        [&collector](const ProgramRealtimeRecordBatch& batch) { collector.consume(batch); });

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    for (uint32_t i = 0; i < kNumProgramsInNonTraceBurst; ++i) {
        distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    }
    distributed::Finish(cq);

    const uint64_t expected_records = static_cast<uint64_t>(kNumProgramsInNonTraceBurst + 1) * rt->num_active_devices();
    const auto wait_result = collector.wait_for_record_count(
        kStressRuntimeId, static_cast<std::size_t>(expected_records), std::chrono::seconds(20));
    UnregisterProgramRealtimeProfilerCallback(handle);

    const auto records = collector.records();
    const auto observed_records = static_cast<uint64_t>(
        std::count_if(records.begin(), records.end(), [](const auto& r) { return r.runtime_id == kStressRuntimeId; }));
    EXPECT_EQ(wait_result.host_dropped, 0u);
    EXPECT_TRUE(wait_result.complete) << "non-trace burst produced " << observed_records << " of " << expected_records
                                      << " expected records";
    EXPECT_EQ(observed_records, expected_records);
    EXPECT_TRUE(mesh_device->close());
}

TEST(RealtimeProfilerStress, DispatchDDoesNotWaitForProfilerWhileDispatchSIsStalled) {
    auto mesh_device = open_full_mesh();
    ASSERT_NE(mesh_device, nullptr);
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    constexpr uint32_t kProgramCount = 64;
    constexpr uint8_t kCqId = 0;
    const auto& mem_map = MetalContext::instance(mesh_device->impl().get_context_id()).dispatch_mem_map();
    DeviceCommandCalculator calculator;
    calculator.add_dispatch_go_signal_mcast();
    for (uint32_t i = 0; i < kProgramCount; ++i) {
        calculator.add_dispatch_set_write_offsets(0);
    }
    calculator.add_notify_dispatch_s_go_signal_cmd();

    auto* device = mesh_device->get_devices().front();
    auto& sysmem_manager = device->sysmem_manager();
    const uint32_t command_bytes = calculator.write_offset_bytes();
    void* command_region = sysmem_manager.issue_queue_reserve(command_bytes, kCqId);
    HugepageDeviceCommand command(command_region, command_bytes);

    // Send a subordinate GO without its dispatch_d notification. dispatch_s blocks on its
    // synchronization semaphore while prefetch continues feeding dispatch_d below.
    command.add_dispatch_go_signal_mcast(
        0,
        MetalContext::instance(mesh_device->impl().get_context_id())
            .hal()
            .make_go_msg_u32(dev_msgs::RUN_MSG_RESET_READ_PTR, 0, 0, 0),
        mem_map.get_dispatch_stream_index(0),
        CQ_DISPATCH_CMD_GO_NO_MULTICAST_OFFSET,
        0,
        0,
        0,
        0,
        DispatcherSelect::DISPATCH_SUBORDINATE);

    const std::array<uint32_t, 0> no_write_offsets{};
    for (uint32_t i = 0; i < kProgramCount; ++i) {
        const uint32_t command_offset = command.write_offset_bytes();
        command.add_dispatch_set_write_offsets(no_write_offsets);
        auto* dispatch_cmd = reinterpret_cast<CQDispatchCmd*>(
            reinterpret_cast<uint8_t*>(command.data()) + command_offset + sizeof(CQPrefetchCmd));
        dispatch_cmd->set_write_offset.program_host_id = static_cast<uint16_t>(i + 1);
    }

    // This command is processed by dispatch_d only after all 64 program IDs. With the old
    // 32-entry profiler FIFO, dispatch_d blocked before reaching it and never released dispatch_s.
    command.add_notify_dispatch_s_go_signal_cmd(false, /*index_bitmask=*/1);
    ASSERT_EQ(command.size_bytes(), command.write_offset_bytes());

    sysmem_manager.issue_queue_push_back(command_bytes, kCqId);
    sysmem_manager.fetch_queue_reserve_back(kCqId);
    sysmem_manager.fetch_queue_write(command_bytes, kCqId);
    distributed::Finish(mesh_device->mesh_command_queue(kCqId));

    EXPECT_TRUE(mesh_device->close());
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
    uint64_t bad_frequency = 0;
    uint64_t invalid_device_intervals = 0;
    uint64_t max_callback_batch = 0;
    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&](const ProgramRealtimeRecordBatch& batch) {
            max_callback_batch = std::max<uint64_t>(max_callback_batch, batch.records.size());
            for (const auto& rec : batch.records) {
                if (rec.runtime_id != kStressRuntimeId) {
                    continue;
                }
                ++stress_records;
                if (!(rec.frequency > 0.0)) {
                    ++bad_frequency;
                }
                if (rec.start_timestamp == 0 || rec.end_timestamp <= rec.start_timestamp) {
                    ++invalid_device_intervals;
                }
            }
        });

    distributed::MeshWorkload workload = build_blank_kernel_workload(mesh_device);
    auto& cq = mesh_device->mesh_command_queue(0);

    // Compile and warm up before the timed ordinary-launch stress loop.
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);

    const std::chrono::seconds stress_window(
        tt::parse_env<std::uint32_t>("TT_RT_PROFILER_SATURATION_SECONDS", kDefaultStressSeconds));
    uint64_t num_stress_programs = 0;
    const auto stress_deadline = std::chrono::steady_clock::now() + stress_window;
    do {
        for (uint32_t i = 0; i < kNumProgramsInNonTraceBurst; ++i) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        distributed::Finish(cq);
        num_stress_programs += kNumProgramsInNonTraceBurst;
    } while (std::chrono::steady_clock::now() < stress_deadline);

    mesh_device->quiesce_devices();
    std::this_thread::sleep_for(kPostQuiesceDrain);
    const uint32_t peak_fifo_pages = rt->peak_fifo_pages();
    const uint32_t fifo_capacity_pages = rt->host_fifo_capacity_pages();
    const uint32_t ring_full_waits = rt->ring_full_wait_count();
    const uint64_t published_batches = rt->num_published_batches();
    const double mean_publish_batch =
        published_batches ? static_cast<double>(rt->num_published_records()) / published_batches : 0.0;
    UnregisterProgramRealtimeProfilerCallback(handle);

    const uint64_t expected_stress_records = num_stress_programs * num_active_devices;

    log_info(
        tt::LogTest,
        "[RT profiler stress] {} stress records across {} active device(s) over {} ordinary launches, "
        "max_callback_batch={}, mean_publish_batch={:.1f}, peak_fifo={}/{} pages, ring_full_waits={}, "
        "{} bad-frequency, {} invalid-device-interval",
        stress_records,
        num_active_devices,
        num_stress_programs,
        max_callback_batch,
        mean_publish_batch,
        peak_fifo_pages,
        fifo_capacity_pages,
        ring_full_waits,
        bad_frequency,
        invalid_device_intervals);

    ASSERT_GE(stress_records, expected_stress_records)
        << "expected one record per ordinary launch: " << num_stress_programs << " launches x " << num_active_devices
        << " active device(s). A shortfall means profiler records were dropped at some point in the pipeline.";

    EXPECT_LT(peak_fifo_pages, fifo_capacity_pages)
        << "host D2H FIFO reached capacity; the receiver drained it slower than the device filled it";

    EXPECT_EQ(ring_full_waits, 0u)
        << "device ring reached capacity; the receiver drained it slower than the device filled it";

    EXPECT_EQ(bad_frequency, 0u) << bad_frequency << " stress record(s) had a non-positive frequency";
    EXPECT_EQ(invalid_device_intervals, 0u)
        << "Every stress record must carry a positive interval measured entirely on device";

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

    constexpr auto kCalibrationWindow = std::chrono::seconds(2);
    const uint64_t pubs_before = rt->num_published_records();
    const auto cal_start = std::chrono::steady_clock::now();
    const auto cal_deadline = cal_start + kCalibrationWindow;
    while (std::chrono::steady_clock::now() < cal_deadline) {
        for (uint32_t i = 0; i < kNumProgramsInNonTraceBurst; ++i) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        distributed::Finish(cq);
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
        for (uint32_t i = 0; i < kNumProgramsInNonTraceBurst; ++i) {
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        distributed::Finish(cq);
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
