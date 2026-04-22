// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merge-gate sanity check for the real-time (RT) profiler on Wormhole and
// Blackhole single-chip configurations. Enqueues a handful of compute
// programs back-to-back on all tensix cores, attaches an RT profiler
// callback, and asserts that each program produces a record with a
// plausible start/end timestamp. The goal is to catch coarse regressions
// in the RT profiler pipeline (mailbox layout, D2H socket init, sync
// handshake, kernel source propagation, timestamp extraction) before they
// reach CI's longer-running profiler test suite.
//
// Lives in the dispatch "basic" test library so it runs as part of
// `tt-metalium-validation-basic`, which the merge-gate `metalium-basic-tests`
// job executes on both N150 (WH) and P150b (BH). On configs where RT
// profiler cannot be enabled (ETH dispatch, non-MMIO chip, kernels
// nullified, IOMMU-off on BH, etc.) the test skips gracefully via
// IsProgramRealtimeProfilerActive().

#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
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

namespace tt::tt_metal {
namespace {

using tt::tt_metal::experimental::IsProgramRealtimeProfilerActive;
using tt::tt_metal::experimental::ProgramRealtimeProfilerCallbackHandle;
using tt::tt_metal::experimental::ProgramRealtimeRecord;
using tt::tt_metal::experimental::RegisterProgramRealtimeProfilerCallback;
using tt::tt_metal::experimental::UnregisterProgramRealtimeProfilerCallback;

constexpr uint32_t kNumPrograms = 5;
// Generous upper bound: the multi_op kernels run ~40K unrolled NOPs. Even
// on slow silicon that stays in the tens-of-microseconds range, so 1s is a
// sanity cap only intended to catch a broken clock / mis-decoded timestamp.
constexpr double kMaxDurationNs = 1'000'000'000.0;

// Runs a single compute program on all tensix cores on `mesh_device`,
// tagged with `runtime_id`, so the RT profiler pipeline emits a record
// with program_id == runtime_id (records with program_id == 0 are filtered
// out by the host-side receiver).
void enqueue_sanity_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t runtime_id, const CoreRange& all_cores) {
    Program program = CreateProgram();

    CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    CreateKernel(
        program,
        "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op_compute.cpp",
        all_cores,
        ComputeConfig{});

    program.set_runtime_id(static_cast<uint64_t>(runtime_id));

    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
}

TEST(RealtimeProfilerSanity, FiveProgramsBackToBack) {
    constexpr int kDeviceId = 0;

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(
        kDeviceId,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1,
        DispatchCoreConfig{DispatchCoreType::WORKER});
    ASSERT_NE(mesh_device, nullptr);

    // Activation flips on during the init-sync handshake inside mesh open,
    // so this check is stable by the time create_unit_mesh returns. When it
    // returns false the RT profiler was disabled for this dispatch config
    // (ETH dispatch, non-MMIO chip, kernels nullified, no valid RT core) —
    // treat that as a graceful skip rather than a failure.
    if (!IsProgramRealtimeProfilerActive()) {
        mesh_device->close();
        GTEST_SKIP() << "Real-time profiler is not active on this dispatch config";
    }

    std::mutex records_mu;
    std::vector<ProgramRealtimeRecord> records;

    ProgramRealtimeProfilerCallbackHandle handle =
        RegisterProgramRealtimeProfilerCallback([&records_mu, &records](const ProgramRealtimeRecord& record) {
            std::lock_guard<std::mutex> lk(records_mu);
            records.push_back(record);
        });

    CoreCoord compute_grid = mesh_device->compute_with_storage_grid_size();
    CoreRange all_cores(CoreCoord{0, 0}, CoreCoord{compute_grid.x - 1, compute_grid.y - 1});

    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        // Runtime IDs start at 1 so every program emits a record (pid == 0
        // is reserved for infrastructure traffic and filtered host-side).
        enqueue_sanity_program(mesh_device, /*runtime_id=*/i + 1, all_cores);
    }

    mesh_device->quiesce_devices();
    // Give the RT profiler receiver thread a moment to drain the last
    // socket pages before we unregister. 500ms mirrors the programming
    // example at test_realtime_profiler_csv.cpp and has proven sufficient
    // for small workloads on WH/BH single-chip.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    UnregisterProgramRealtimeProfilerCallback(handle);

    std::vector<ProgramRealtimeRecord> collected;
    {
        std::lock_guard<std::mutex> lk(records_mu);
        collected = std::move(records);
    }

    ASSERT_GE(collected.size(), kNumPrograms)
        << "Expected at least " << kNumPrograms << " RT profiler records (one per program), got " << collected.size();

    for (const auto& rec : collected) {
        EXPECT_GT(rec.end_timestamp, rec.start_timestamp)
            << "RT record end_timestamp must be strictly greater than start_timestamp (pid=" << rec.program_id
            << ", chip=" << rec.chip_id << ")";
        EXPECT_GT(rec.frequency, 0.0) << "RT record frequency must be positive (pid=" << rec.program_id
                                      << ", chip=" << rec.chip_id << ")";

        if (rec.frequency > 0.0 && rec.end_timestamp > rec.start_timestamp) {
            uint64_t duration_cycles = rec.end_timestamp - rec.start_timestamp;
            double duration_ns = static_cast<double>(duration_cycles) / rec.frequency;
            EXPECT_LT(duration_ns, kMaxDurationNs)
                << "RT record duration is implausibly large (pid=" << rec.program_id << ", chip=" << rec.chip_id
                << ", duration_ns=" << duration_ns << ")";
        }
    }

    EXPECT_TRUE(mesh_device->close());
}

}  // namespace
}  // namespace tt::tt_metal
