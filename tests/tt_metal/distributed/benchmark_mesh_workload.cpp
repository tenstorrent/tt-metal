// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host cost of EnqueueMeshWorkload on the whole mesh, for workload shapes taken from models. While dispatch is
// host-bound, the time per call is the gap between ops.
//
// Glm52 follows GLM-5.2 chunked prefill on an 8x4 Blackhole Galaxy, measured with TT_METAL_DISPATCH_STATS
// (#57586). Every enqueue covers the whole mesh, as one program over all devices (63%) or one program per device
// (35%), with 6.1 KB of command stream per device at the median and 14 KB at p90. The host spends 63 us between
// enqueues at the median. Kimi K2.7 looks the same. Command bytes depend on the device's core grid, so shapes name
// a target size per device, and the runtime-arg count that reaches it is found on the device at hand.
//
// The kernels do no work, so the device keeps up with the host. Several distinct workloads are cycled so that each
// call dispatches a program the host has not just dispatched, and like TTNN, every dispatch gets a fresh runtime id.

#include <benchmark/benchmark.h>

#include <sys/resource.h>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include <algorithm>
#include <chrono>
#include <map>
#include <tuple>
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

// Workloads with the same program layout. target_bytes_per_device sets the runtime args so that one enqueue writes
// about that many command bytes to each device; 0 uses the shape's runtime_args.
struct WorkloadPart {
    // Programs cover the mesh in ranges of at most this many devices along its longer axis; 0 for one program
    // over the whole mesh.
    uint32_t max_devices_per_program;
    uint32_t target_bytes_per_device;
    uint32_t count;
};

struct WorkloadShape {
    const char* name;
    std::vector<WorkloadPart> parts;
    uint32_t num_cbs;
    // Unique runtime args per kernel per core, for parts without a target size.
    uint32_t runtime_args;
    // Host time spent between calls, standing in for the model's work between enqueues.
    uint32_t gap_us;
    int64_t iterations;
};

constexpr const char* KERNEL =
    "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/command_queue/random_program.cpp";
// CB i holds one page of (i + 1) * CB_PAGE_SIZE bytes, which the kernel checks.
constexpr uint32_t CB_PAGE_SIZE = 64;

Program make_program(uint32_t num_cbs, uint32_t num_runtime_args, const CoreRangeSet& cores) {
    Program program;
    for (uint32_t i = 0; i < num_cbs; i++) {
        const uint32_t size = (i + 1) * CB_PAGE_SIZE;
        CreateCircularBuffer(
            program, cores, CircularBufferConfig(size, {{i, tt::DataFormat::Float16_b}}).set_page_size(i, size));
    }
    // The kernel expects unique runtime arg i to be i.
    std::vector<uint32_t> runtime_args(num_runtime_args);
    std::iota(runtime_args.begin(), runtime_args.end(), 0);
    // One iteration of each loop, no semaphores, no common runtime args.
    const std::vector<uint32_t> compile_args = {1, 1, 1, num_cbs, 0, num_runtime_args, 0, CB_PAGE_SIZE};
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

std::unique_ptr<MeshWorkload> make_workload(
    const std::vector<MeshCoordinateRange>& ranges,
    uint32_t num_cbs,
    uint32_t num_runtime_args,
    const CoreRangeSet& cores) {
    auto workload = std::make_unique<MeshWorkload>();
    for (const auto& range : ranges) {
        workload->add_program(range, make_program(num_cbs, num_runtime_args, cores));
    }
    return workload;
}

// Command bytes one re-enqueue of the workload writes to the device at the start of the first range.
uint32_t cmd_bytes_per_device(MeshWorkload& workload, const MeshCoordinateRange& first_range) {
    auto& cq = mesh_device().mesh_command_queue();
    auto& sysmem = mesh_device().get_device(first_range.start_coord())->sysmem_manager();
    const auto cq_id = static_cast<uint8_t>(cq.id());
    // The first enqueue also writes the binaries.
    EnqueueMeshWorkload(cq, workload, false);
    Finish(cq);
    for (int attempt = 0; attempt < 3; attempt++) {
        const uint32_t before = sysmem.get_issue_queue_write_ptr(cq_id);
        EnqueueMeshWorkload(cq, workload, false);
        const uint32_t after = sysmem.get_issue_queue_write_ptr(cq_id);
        Finish(cq);
        if (after > before) {
            return after - before;
        }
    }
    return 0;
}

// Fewest runtime args per kernel per core for which an enqueue writes at least `target` bytes to each device.
uint32_t runtime_args_for_bytes(
    const std::vector<MeshCoordinateRange>& ranges, uint32_t num_cbs, uint32_t target, const CoreRangeSet& cores) {
    static std::map<std::tuple<size_t, uint32_t, uint32_t>, uint32_t> cache;
    const auto key = std::make_tuple(ranges.size(), num_cbs, target);
    if (auto it = cache.find(key); it != cache.end()) {
        return it->second;
    }
    auto bytes_with = [&](uint32_t num_runtime_args) {
        auto workload = make_workload(ranges, num_cbs, num_runtime_args, cores);
        return cmd_bytes_per_device(*workload, ranges.front());
    };
    uint32_t lo = 0;
    uint32_t hi = max_runtime_args;
    if (bytes_with(hi) < target) {
        lo = hi;
    }
    while (lo < hi) {
        const uint32_t mid = (lo + hi) / 2;
        if (bytes_with(mid) >= target) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return cache[key] = lo;
}

void spin_for_us(uint32_t us) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(us);
    while (std::chrono::steady_clock::now() < end) {
    }
}

double process_cpu_seconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_utime.tv_sec + (ru.ru_utime.tv_usec * 1e-6) + ru.ru_stime.tv_sec + (ru.ru_stime.tv_usec * 1e-6);
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
    // Each part's workloads are spread evenly through the cycle.
    std::vector<std::unique_ptr<MeshWorkload>> workloads;
    std::vector<std::vector<MeshCoordinateRange>> part_ranges;
    std::vector<uint32_t> part_runtime_args;
    uint32_t total = 0;
    for (const auto& part : shape.parts) {
        part_ranges.push_back(split_mesh(part.max_devices_per_program));
        part_runtime_args.push_back(
            part.target_bytes_per_device == 0
                ? shape.runtime_args
                : runtime_args_for_bytes(part_ranges.back(), shape.num_cbs, part.target_bytes_per_device, cores));
        total += part.count;
    }
    std::vector<uint32_t> emitted(shape.parts.size(), 0);
    for (uint32_t w = 0; w < total; w++) {
        size_t pick = 0;
        double best = -1;
        for (size_t p = 0; p < shape.parts.size(); p++) {
            const double owed = (static_cast<double>(shape.parts[p].count) * (w + 1) / total) - emitted[p];
            if (emitted[p] < shape.parts[p].count && owed > best) {
                best = owed;
                pick = p;
            }
        }
        emitted[pick]++;
        workloads.push_back(make_workload(part_ranges[pick], shape.num_cbs, part_runtime_args[pick], cores));
    }
    const uint32_t num_workloads = total;
    auto& cq = mesh_device().mesh_command_queue();
    // The first enqueue of each workload compiles it and writes its binaries.
    for (auto& workload : workloads) {
        EnqueueMeshWorkload(cq, *workload, false);
    }
    Finish(cq);

    // Command bytes written to one device per call; calls where the issue queue wrapped are skipped.
    auto& sysmem = mesh_device().get_device(part_ranges.front().front().start_coord())->sysmem_manager();
    const auto cq_id = static_cast<uint8_t>(cq.id());
    std::vector<double> call_us, bytes_per_device, programs;
    uint32_t next = 0;
    const double cpu_before = process_cpu_seconds();
    const auto wall_before = std::chrono::steady_clock::now();
    ProgramId runtime_id = 1;
    for ([[maybe_unused]] auto _ : state) {
        auto& workload = *workloads[next++ % num_workloads];
        for (auto& [range, program] : workload.get_programs()) {
            program.set_runtime_id(runtime_id++);
        }
        spin_for_us(shape.gap_us);
        const uint32_t wptr_before = sysmem.get_issue_queue_write_ptr(cq_id);
        const auto t0 = std::chrono::steady_clock::now();
        EnqueueMeshWorkload(cq, workload, false);
        const auto t1 = std::chrono::steady_clock::now();
        const uint32_t wptr_after = sysmem.get_issue_queue_write_ptr(cq_id);
        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        state.SetIterationTime(seconds);
        call_us.push_back(seconds * 1e6);
        programs.push_back(static_cast<double>(workload.get_programs().size()));
        if (wptr_after > wptr_before) {
            bytes_per_device.push_back(wptr_after - wptr_before);
        }
    }
    const double cpu_s = process_cpu_seconds() - cpu_before;
    const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_before).count();
    Finish(cq);

    // Host CPU the whole process used while the calls ran, including any thread pool workers polling for work.
    state.counters["process_cpu_cores"] = cpu_s / wall_s;
    state.counters["p50_us"] = percentile(call_us, 0.5);
    state.counters["p99_us"] = percentile(call_us, 0.99);
    state.counters["programs"] = std::accumulate(programs.begin(), programs.end(), 0.0) / programs.size();
    state.counters["devices"] = static_cast<double>(mesh_device().num_devices());
    if (!bytes_per_device.empty()) {
        state.counters["cmd_bytes_per_device"] = percentile(bytes_per_device, 0.5);
        state.counters["cmd_bytes_per_device_p90"] = percentile(bytes_per_device, 0.9);
    }
}

const bool registered = [] {
    // GLM-5.2's mix of layouts and sizes, with 16 workloads in the cycle (#57586).
    const std::vector<WorkloadPart> glm52 = {
        {.max_devices_per_program = 0, .target_bytes_per_device = 6144, .count = 8},
        {.max_devices_per_program = 0, .target_bytes_per_device = 14336, .count = 2},
        {.max_devices_per_program = 1, .target_bytes_per_device = 6144, .count = 6},
    };
    static const WorkloadShape shapes[] = {
        {.name = "Glm52", .parts = glm52, .num_cbs = 4, .runtime_args = 0, .gap_us = 0, .iterations = 5000},
        // With the median host time between GLM's enqueues, so thread pool workers idle as they do in the model.
        {.name = "Glm52Gap63us", .parts = glm52, .num_cbs = 4, .runtime_args = 0, .gap_us = 63, .iterations = 5000},
        {.name = "OneProgram",
         .parts = {{.max_devices_per_program = 0, .target_bytes_per_device = 6144, .count = 8}},
         .num_cbs = 4,
         .runtime_args = 0,
         .gap_us = 0,
         .iterations = 5000},
        {.name = "ProgramPerDevice",
         .parts = {{.max_devices_per_program = 1, .target_bytes_per_device = 6144, .count = 8}},
         .num_cbs = 4,
         .runtime_args = 0,
         .gap_us = 0,
         .iterations = 5000},
        {.name = "Small",
         .parts = {{.max_devices_per_program = 0, .target_bytes_per_device = 0, .count = 8}},
         .num_cbs = 1,
         .runtime_args = 0,
         .gap_us = 0,
         .iterations = 5000},
        // The runtime-arg-heavy shape #52775 was tuned on; no model op looks like this today.
        {.name = "BytesHeavy",
         .parts = {{.max_devices_per_program = 0, .target_bytes_per_device = 0, .count = 8}},
         .num_cbs = 12,
         .runtime_args = max_runtime_args,
         .gap_us = 0,
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
