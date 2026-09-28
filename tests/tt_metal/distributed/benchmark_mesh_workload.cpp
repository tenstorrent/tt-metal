// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host cost of EnqueueMeshWorkload on the whole mesh, for workload shapes taken from models. While dispatch is
// host-bound, the time per call is the gap between ops.
//
// Glm follows an instrumented GLM 5.2 chunked-prefill run on an 8x4 Blackhole Galaxy (#53562, #54992): each
// enqueue carries ~11 programs of ~3 devices each, one program per device, with ~9 KB of command stream per
// device. The kernels do no work, so the device keeps up with the host. Several distinct workloads are cycled so
// that each call dispatches a program the host has not just dispatched, and like TTNN, every dispatch gets a fresh
// runtime id.

#include <benchmark/benchmark.h>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "benchmark_mesh_device.hpp"
#include "tt_metal/impl/dispatch/system_memory_manager.hpp"

namespace {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

MeshDevice& mesh_device() { return test::benchmark_mesh_device(); }

struct WorkloadShape {
    const char* name;
    // Programs cover the mesh in ranges of at most this many devices along its longer axis; 0 for one program
    // over the whole mesh.
    uint32_t max_devices_per_program;
    uint32_t num_cbs;
    // Unique runtime args per kernel per core.
    uint32_t runtime_args;
    int64_t iterations;
};

constexpr const char* KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp";
// CB i holds one page of (i + 1) * CB_PAGE_SIZE bytes, which the kernel checks.
constexpr uint32_t CB_PAGE_SIZE = 64;

Program make_program(const WorkloadShape& shape, const CoreRangeSet& cores) {
    Program program;
    for (uint32_t i = 0; i < shape.num_cbs; i++) {
        const uint32_t size = (i + 1) * CB_PAGE_SIZE;
        CreateCircularBuffer(
            program, cores, CircularBufferConfig(size, {{i, tt::DataFormat::Float16_b}}).set_page_size(i, size));
    }
    // The kernel expects unique runtime arg i to be i.
    std::vector<uint32_t> runtime_args(shape.runtime_args);
    std::iota(runtime_args.begin(), runtime_args.end(), 0);
    // One iteration of each loop, no semaphores, no common runtime args.
    const std::vector<uint32_t> compile_args = {1, 1, 1, shape.num_cbs, 0, shape.runtime_args, 0, CB_PAGE_SIZE};
    const std::map<std::string, std::string> data_movement = {{"DATA_MOVEMENT", "1"}};
    const std::map<std::string, std::string> compute = {{"COMPUTE", "1"}};
    const auto kernels = {
        CreateKernel(
            program,
            KERNEL,
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = compile_args,
                .defines = data_movement}),
        CreateKernel(
            program,
            KERNEL,
            cores,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = compile_args,
                .defines = data_movement}),
        CreateKernel(program, KERNEL, cores, ComputeConfig{.compile_args = compile_args, .defines = compute}),
    };
    for (auto kernel : kernels) {
        SetRuntimeArgs(program, kernel, cores, runtime_args);
    }
    return program;
}

std::vector<MeshCoordinateRange> split_mesh(uint32_t max_devices_per_program) {
    const auto& shape = mesh_device().shape();
    if (max_devices_per_program == 0) {
        return {MeshCoordinateRange(shape)};
    }
    const bool split_cols = shape[1] >= shape[0];
    const uint32_t lines = split_cols ? shape[0] : shape[1];
    const uint32_t length = split_cols ? shape[1] : shape[0];
    std::vector<MeshCoordinateRange> ranges;
    for (uint32_t line = 0; line < lines; line++) {
        for (uint32_t begin = 0; begin < length; begin += max_devices_per_program) {
            const uint32_t end = std::min(begin + max_devices_per_program, length) - 1;
            ranges.push_back(
                split_cols ? MeshCoordinateRange(MeshCoordinate(line, begin), MeshCoordinate(line, end))
                           : MeshCoordinateRange(MeshCoordinate(begin, line), MeshCoordinate(end, line)));
        }
    }
    return ranges;
}

double percentile(std::vector<double> v, double p) {
    auto idx = std::min(v.size() - 1, static_cast<size_t>(std::lround(p * (v.size() - 1))));
    std::nth_element(v.begin(), v.begin() + idx, v.end());
    return v[idx];
}

void BM_EnqueueMeshWorkload(benchmark::State& state, const WorkloadShape& shape) {
    if (mesh_device().shape().dims() != 2) {
        state.SkipWithError("needs a 2D mesh");
        return;
    }
    const CoreCoord grid = mesh_device().compute_with_storage_grid_size();
    const CoreRangeSet cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const auto ranges = split_mesh(shape.max_devices_per_program);

    constexpr uint32_t NUM_WORKLOADS = 8;
    std::vector<std::unique_ptr<MeshWorkload>> workloads;
    for (uint32_t w = 0; w < NUM_WORKLOADS; w++) {
        auto& workload = *workloads.emplace_back(std::make_unique<MeshWorkload>());
        for (const auto& range : ranges) {
            workload.add_program(range, make_program(shape, cores));
        }
    }
    auto& cq = mesh_device().mesh_command_queue();
    // The first enqueue of each workload compiles it and writes its binaries.
    for (auto& workload : workloads) {
        EnqueueMeshWorkload(cq, *workload, false);
    }
    Finish(cq);

    // Command bytes written to one device per call; calls where the issue queue wrapped are skipped.
    auto& sysmem = mesh_device().get_device(ranges.front().start_coord())->sysmem_manager();
    const auto cq_id = static_cast<uint8_t>(cq.id());
    std::vector<double> call_us, bytes_per_device;
    uint32_t next = 0;
    ProgramId runtime_id = 1;
    for ([[maybe_unused]] auto _ : state) {
        auto& workload = *workloads[next++ % NUM_WORKLOADS];
        for (auto& [range, program] : workload.get_programs()) {
            program.set_runtime_id(runtime_id++);
        }
        const uint32_t wptr_before = sysmem.get_issue_queue_write_ptr(cq_id);
        const auto t0 = std::chrono::steady_clock::now();
        EnqueueMeshWorkload(cq, workload, false);
        const auto t1 = std::chrono::steady_clock::now();
        const uint32_t wptr_after = sysmem.get_issue_queue_write_ptr(cq_id);
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        state.SetIterationTime(seconds);
        call_us.push_back(seconds * 1e6);
        if (wptr_after > wptr_before) {
            bytes_per_device.push_back(wptr_after - wptr_before);
        }
    }
    Finish(cq);

    state.counters["p50_us"] = percentile(call_us, 0.5);
    state.counters["p99_us"] = percentile(call_us, 0.99);
    state.counters["programs"] = static_cast<double>(ranges.size());
    state.counters["devices"] = static_cast<double>(mesh_device().num_devices());
    if (!bytes_per_device.empty()) {
        state.counters["cmd_bytes_per_device"] = percentile(bytes_per_device, 0.5);
    }
}

const bool registered = [] {
    static const WorkloadShape shapes[] = {
        {.name = "Glm", .max_devices_per_program = 3, .num_cbs = 12, .runtime_args = 9, .iterations = 5000},
        {.name = "OneProgram", .max_devices_per_program = 0, .num_cbs = 12, .runtime_args = 9, .iterations = 5000},
        {.name = "ProgramPerDevice",
         .max_devices_per_program = 1,
         .num_cbs = 12,
         .runtime_args = 9,
         .iterations = 5000},
        {.name = "Small", .max_devices_per_program = 0, .num_cbs = 1, .runtime_args = 0, .iterations = 5000},
        // The runtime-arg-heavy shape #52775 was tuned on; no model op looks like this today.
        {.name = "BytesHeavy",
         .max_devices_per_program = 0,
         .num_cbs = 12,
         .runtime_args = max_runtime_args,
         .iterations = 1000},
    };
    for (const auto& shape : shapes) {
        benchmark::RegisterBenchmark(std::string("BM_EnqueueMeshWorkload/") + shape.name, BM_EnqueueMeshWorkload, shape)
            ->Iterations(shape.iterations)
            ->UseManualTime()
            ->Unit(benchmark::kMicrosecond);
    }
    return true;
}();

}  // namespace
