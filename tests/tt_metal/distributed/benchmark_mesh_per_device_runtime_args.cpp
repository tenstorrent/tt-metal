// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host enqueue cost of a program whose runtime args differ per device, on the whole mesh:
// - OneProgram/fields=0: one program on the mesh range, no per-device runtime args (the no-field path).
// - OneProgram/fields=30: the same program with 30 per-device fields of 16 words (15 cores x 2 kernels).
// - ProgramPerDevice: the pattern per-device runtime args replace, one program per device.
// Each iteration enqueues the workload once (non-blocking); the queue is drained outside the timed region.

#include <benchmark/benchmark.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

MeshDevice& mesh_device() {
    static std::shared_ptr<MeshDevice> device = MeshDevice::create(MeshDeviceConfig(std::nullopt));
    return *device;
}

constexpr uint32_t kArgs = 16;
const CoreRange kCores({0, 0}, {4, 2});  // 15 cores
constexpr int kDrainEvery = 32;

// Two kernels on kCores, kArgs unique runtime args on every core; returns the kernel handles.
std::vector<KernelHandle> add_kernels(Program& program) {
    const uint32_t results = mesh_device().allocator()->get_base_allocator_addr(HalMemType::L1);
    std::vector<KernelHandle> kernels;
    for (auto processor : {DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1}) {
        const uint32_t addr = results + (processor == DataMovementProcessor::RISCV_0 ? 0 : 4096);
        kernels.push_back(CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/misc/runtime_args_kernel.cpp",
            CoreRangeSet(kCores),
            DataMovementConfig{
                .processor = processor,
                .noc = processor == DataMovementProcessor::RISCV_0 ? NOC::RISCV_0_default : NOC::RISCV_1_default,
                .defines = {{"RESULTS_ADDR", std::to_string(addr)}, {"NUM_RUNTIME_ARGS", std::to_string(kArgs)}}}));
        for (const auto& core : kCores) {
            SetRuntimeArgs(program, kernels.back(), core, std::vector<uint32_t>(kArgs, 1));
        }
    }
    return kernels;
}

void run(benchmark::State& state, MeshWorkload& workload) {
    auto& cq = mesh_device().mesh_command_queue();
    EnqueueMeshWorkload(cq, workload, false);  // compile, load, assemble
    Finish(cq);
    int since_drain = 0;
    for ([[maybe_unused]] auto _ : state) {
        EnqueueMeshWorkload(cq, workload, false);
        if (++since_drain == kDrainEvery) {
            state.PauseTiming();
            Finish(cq);
            since_drain = 0;
            state.ResumeTiming();
        }
    }
    Finish(cq);
}

void BM_OneProgram(benchmark::State& state) {
    const bool with_fields = state.range(0) != 0;
    const MeshCoordinateRange range(mesh_device().shape());
    Program program = CreateProgram();
    const auto kernels = add_kernels(program);
    MeshWorkload workload;
    workload.add_program(range, std::move(program));
    if (with_fields) {
        const uint32_t num_devices = range.shape().mesh_size();
        std::vector<uint32_t> values(num_devices * kArgs);
        for (uint32_t i = 0; i < values.size(); i++) {
            values[i] = i;
        }
        for (const auto kernel : kernels) {
            for (const auto& core : kCores) {
                SetRuntimeArgsPerDevice(workload, range, kernel, core, 0, kArgs, values);
            }
        }
    }
    run(state, workload);
}

void BM_ProgramPerDevice(benchmark::State& state) {
    MeshWorkload workload;
    for (const auto& coord : MeshCoordinateRange(mesh_device().shape())) {
        Program program = CreateProgram();
        add_kernels(program);
        workload.add_program(MeshCoordinateRange(coord), std::move(program));
    }
    run(state, workload);
}

BENCHMARK(BM_OneProgram)
    ->Name("BM_PerDeviceRuntimeArgs/OneProgram")
    ->ArgName("fields")
    ->Arg(0)
    ->Arg(30)
    ->Iterations(4000)
    ->Repetitions(3)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ProgramPerDevice)
    ->Name("BM_PerDeviceRuntimeArgs/ProgramPerDevice")
    ->Iterations(4000)
    ->Repetitions(3)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace
