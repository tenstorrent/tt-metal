// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//////////////////////////////////////////////////////////////////////////////////////////
// Device profiler overhead microbenchmark (#46305)
//
// The device profiler is not free: when it is enabled (compile-time TRACY_ENABLE + runtime
// TT_METAL_DEVICE_PROFILER=1) every kernel launch emits firmware {BRISC,NCRISC,TRISC}-KERNEL
// timestamp zones into per-core L1, and the host must periodically drain those markers to
// DRAM/CSV via ReadMeshDeviceProfilerResults. That drain is the dominant profiler cost a user
// pays per program (reviewer feedback on the op-to-op benchmark, #49771: "op-to-op latency is
// dominated by the time it takes the profiler to write out the profiler buffer"). This
// benchmark isolates and tracks that cost over time so a regression in the profiler read path
// (or in profiler-enabled dispatch) is caught in CI.
//
// Two host-timed families, each over a "common" (single core, one RISC) and a "worst" (full
// worker grid, all five RISCs -> maximum firmware KERNEL zones) program shape:
//
//   * profiler_read     - time ReadMeshDeviceProfilerResults() for a freshly populated buffer.
//                         The program is re-dispatched (untimed) before each read so the buffer
//                         holds one launch worth of markers; only the drain is timed. This is
//                         the profiler-buffer writeout overhead.
//   * profiled_dispatch - time EnqueueMeshWorkload + Finish of a profiler-instrumented program
//                         (kernels compiled with -DPROFILE_KERNEL). Tracks the on-dispatch cost
//                         the profiler adds to the kernel launch path.
//
// Requires a Tracy-enabled (ENABLE_TRACY=ON) build; TT_METAL_DEVICE_PROFILER=1 is forced on in
// main() before device init so kernels JIT-compile with profiler markers and the read path is
// active. Runs deliberately WITHOUT TT_METAL_PROFILER_ACCUMULATE so each read performs the full
// L1->DRAM writeout (the pessimistic, most informative overhead), unlike the op-to-op benchmark
// which accumulates to keep the dump out of the measured op2op gap.
//
// Timing note: benchmarks use UseManualTime() and time only the profiler read / enqueue call;
// buffer re-population and Finish() drains run outside the timed region.
//////////////////////////////////////////////////////////////////////////////////////////

#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-logger/tt-logger.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 20;

// The profiler read (L1->DRAM writeout + CSV post-process) is a slow, mildly jittery host op, so
// each read benchmark runs several repetitions and reports the MIN: a contended host only inflates
// a sample, so the minimum is the most representative measure of true overhead, while a genuine
// regression still raises it. The compare script consumes the "min" aggregate. Repetitions are
// kept small because Google Benchmark re-runs the whole registered function (device open/close
// included) per repetition, and device init on silicon is not cheap.
constexpr uint32_t READ_REPETITIONS = 3;
constexpr uint32_t READ_ITERATIONS = 20;

// Profiler-enabled dispatch is us-scale and cheaper than a read, so it can afford more iterations.
constexpr uint32_t DISPATCH_REPETITIONS = 3;
constexpr uint32_t DISPATCH_ITERATIONS = 200;

constexpr const char* kBlankDataflowKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp";
constexpr const char* kBlankComputeKernel = "tests/tt_metal/tt_metal/test_kernels/compute/blank.cpp";

// A program shape: which worker cores run kernels and whether all five RISCs are populated. More
// cores x more RISCs => more firmware KERNEL zones the profiler must write out per launch.
struct ProgramShape {
    std::string name;
    bool full_grid = false;  // false: single core (0,0); true: full compute-with-storage grid
    bool all_riscs = false;  // false: BRISC only; true: BRISC + NCRISC + 3x TRISC (compute)
};

const ProgramShape COMMON_SHAPE{.name = "common", .full_grid = false, .all_riscs = false};
const ProgramShape WORST_SHAPE{.name = "worst", .full_grid = true, .all_riscs = true};

std::shared_ptr<MeshDevice> open_device() {
    constexpr ChipId device_id = 0;
    return MeshDevice::create_unit_mesh(device_id);
}

CoreRange shape_grid(const MeshDevice& device, const ProgramShape& shape) {
    if (!shape.full_grid) {
        return CoreRange({0, 0}, {0, 0});
    }
    auto grid = device.compute_with_storage_grid_size();
    return CoreRange({0, 0}, {grid.x - 1, grid.y - 1});
}

// Build a program of blank kernels over `shape`'s grid. When profiling is enabled the kernels are
// JIT-compiled with -DPROFILE_KERNEL, so firmware emits per-core KERNEL timestamp zones at launch.
Program build_program(const MeshDevice& device, const ProgramShape& shape) {
    Program program = CreateProgram();
    const CoreRange grid = shape_grid(device, shape);

    CreateKernel(
        program,
        kBlankDataflowKernel,
        grid,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    if (shape.all_riscs) {
        CreateKernel(
            program,
            kBlankDataflowKernel,
            grid,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        CreateKernel(program, kBlankComputeKernel, grid, ComputeConfig{});
    }
    return program;
}

MeshWorkload make_workload(Program&& program) {
    MeshWorkload workload;
    workload.add_program(MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)), std::move(program));
    return workload;
}

//////////////////////////////////////////////////////////////////////////////////////////
// profiler_read: time ReadMeshDeviceProfilerResults() draining one launch worth of markers.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_profiler_read(benchmark::State& state, const ProgramShape& shape) {
    auto device = open_device();
    auto& cq = device->mesh_command_queue();
    MeshWorkload workload = make_workload(build_program(*device, shape));

    for (uint32_t i = 0; i < DEFAULT_WARMUP_ITERATIONS; i++) {
        EnqueueMeshWorkload(cq, workload, false);
    }
    Finish(cq);
    ReadMeshDeviceProfilerResults(*device);  // drain warmup markers

    for ([[maybe_unused]] auto _ : state) {
        // Repopulate the profiler buffer with a single launch (untimed).
        EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);

        auto start = std::chrono::steady_clock::now();
        ReadMeshDeviceProfilerResults(*device);
        auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    }
    device->close();
}

//////////////////////////////////////////////////////////////////////////////////////////
// profiled_dispatch: time EnqueueMeshWorkload + Finish of a profiler-instrumented program.
//////////////////////////////////////////////////////////////////////////////////////////
void BM_profiled_dispatch(benchmark::State& state, const ProgramShape& shape) {
    auto device = open_device();
    auto& cq = device->mesh_command_queue();
    MeshWorkload workload = make_workload(build_program(*device, shape));

    for (uint32_t i = 0; i < DEFAULT_WARMUP_ITERATIONS; i++) {
        EnqueueMeshWorkload(cq, workload, false);
    }
    Finish(cq);

    // The device-side profiler buffer is left to overwrite in place across iterations (no host
    // drain): we only measure the dispatch cost the profiler adds, not the readout.
    for ([[maybe_unused]] auto _ : state) {
        auto start = std::chrono::steady_clock::now();
        EnqueueMeshWorkload(cq, workload, false);
        Finish(cq);
        auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count());
    }
    device->close();
}

// Repetitions + MIN aggregate config, applied at registration.
void ReadConfig(benchmark::internal::Benchmark* b) {
    b->Iterations(READ_ITERATIONS)
        ->Repetitions(READ_REPETITIONS)
        ->ComputeStatistics("min", [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); })
        ->ReportAggregatesOnly(true)
        ->UseManualTime();
}

void DispatchConfig(benchmark::internal::Benchmark* b) {
    b->Iterations(DISPATCH_ITERATIONS)
        ->Repetitions(DISPATCH_REPETITIONS)
        ->ComputeStatistics("min", [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); })
        ->ReportAggregatesOnly(true)
        ->UseManualTime();
}

}  // namespace

BENCHMARK_CAPTURE(BM_profiler_read, common, COMMON_SHAPE)->Apply(ReadConfig);
BENCHMARK_CAPTURE(BM_profiler_read, worst, WORST_SHAPE)->Apply(ReadConfig);
BENCHMARK_CAPTURE(BM_profiled_dispatch, common, COMMON_SHAPE)->Apply(DispatchConfig);
BENCHMARK_CAPTURE(BM_profiled_dispatch, worst, WORST_SHAPE)->Apply(DispatchConfig);

int main(int argc, char** argv) {
#if defined(TRACY_ENABLE)
    // Force the benchmark's required runtime mode before any MetalContext / rtoptions construction,
    // OVERWRITING any inherited value: kernels must JIT-compile with -DPROFILE_KERNEL and the read
    // path must be active, so an inherited TT_METAL_DEVICE_PROFILER=0 would silently invalidate the
    // measurement. Likewise clear TT_METAL_PROFILER_ACCUMULATE so each read performs the full
    // L1->DRAM writeout (this benchmark's contract) rather than the accumulate read path.
    setenv("TT_METAL_DEVICE_PROFILER", "1", /*overwrite=*/1);
    unsetenv("TT_METAL_PROFILER_ACCUMULATE");
#else
    // Without Tracy the profiler is compiled out: ReadMeshDeviceProfilerResults is a no-op and the
    // measured overhead is ~0. Emitting near-zero results with exit code 0 would let a misconfigured
    // (non-profiler) build pass CI, or read as an improvement under a future one-sided gate. Fail
    // hard instead so the benchmark is only ever run against a meaningful build.
    log_error(tt::LogTest, "benchmark_profiler_overhead requires a Tracy-enabled build; rebuild with ENABLE_TRACY=ON.");
    return 1;
#endif

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
