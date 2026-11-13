// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host enqueue cost of a program whose runtime args differ per device, on the whole mesh:
// - OneProgram/fields=0: one program on the mesh range, no per-device runtime args (the no-field path).
// - OneProgram/fields=30: the same program with 30 per-device fields of 16 words (15 cores x 2 kernels).
// - OneProgram/fields=30/reset: as fields=30, and every iteration sets the same per-device values again first (an op's
//   cache hit): the per-device command images must be kept. reset=2: different values every iteration, so every
//   enqueue rebuilds the per-device command images (what reset=1 cost before same values were detected).
// - ProgramPerDevice: the pattern per-device runtime args replace, one program per device.
// - Metal2/...: a Metal 2.0 program on the mesh with 30 named common run args; every iteration calls
//   UpdateProgramRunArgs first. mode=0: no per-device values, update "address". mode=1: the 30 args have per-device
//   values, update "address" (no drop). mode=2: update the 30 args (dropping their per-device values), then set the
//   same per-device values again on every device. mode=3: only set the same per-device values again (no update).
// Each iteration enqueues the workload once (non-blocking); the queue is drained outside the timed region.

#include <benchmark/benchmark.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "tests/tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"

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

template <typename BeforeEnqueue = void (*)()>
void run(benchmark::State& state, MeshWorkload& workload, BeforeEnqueue before_enqueue = [] {}) {
    auto& cq = mesh_device().mesh_command_queue();
    before_enqueue();
    EnqueueMeshWorkload(cq, workload, false);  // compile, load, assemble
    Finish(cq);
    int since_drain = 0;
    for ([[maybe_unused]] auto _ : state) {
        before_enqueue();
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
    const int64_t reset = state.range(1);
    const MeshCoordinateRange range(mesh_device().shape());
    Program program = CreateProgram();
    const auto kernels = add_kernels(program);
    MeshWorkload workload;
    workload.add_program(range, std::move(program));
    const uint32_t num_devices = range.shape().mesh_size();
    std::vector<uint32_t> values(num_devices * kArgs);
    for (uint32_t i = 0; i < values.size(); i++) {
        values[i] = i;
    }
    auto set_fields = [&] {
        if (reset == 2) {
            values[0] ^= 1;
        }
        for (const auto kernel : kernels) {
            for (const auto& core : kCores) {
                SetRuntimeArgsPerDevice(workload, range, kernel, core, 0, kArgs, values);
            }
        }
    };
    if (with_fields) {
        set_fields();
    }
    if (reset != 0) {
        run(state, workload, set_fields);
    } else {
        run(state, workload);
    }
}

namespace m2 = tt::tt_metal::experimental;
constexpr uint32_t kMetal2Args = 30;
constexpr char kMetal2Kernel[] = "writer";

m2::ProgramRunArgs metal2_args(bool address, bool values, uint32_t base) {
    const uint32_t results = mesh_device().allocator()->get_base_allocator_addr(HalMemType::L1);
    m2::ProgramRunArgs::KernelRunArgs kernel_args{.kernel = m2::KernelSpecName{kMetal2Kernel}};
    if (address) {
        kernel_args.runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(m2::NodeCoord{0, 0}, {{"address", results}});
    }
    if (values) {
        for (uint32_t i = 0; i < kMetal2Args; i++) {
            kernel_args.common_runtime_arg_values["v" + std::to_string(i)] = base + i;
        }
    }
    m2::ProgramRunArgs args;
    args.kernel_run_args = {std::move(kernel_args)};
    return args;
}

void BM_Metal2(benchmark::State& state) {
    const int64_t mode = state.range(0);
    auto kernel = m2::test_helpers::MakeMinimalGen1DMKernel(kMetal2Kernel, DataMovementProcessor::RISCV_0);
    kernel.runtime_arg_schema.runtime_arg_names = {"address"};
    for (uint32_t i = 0; i < kMetal2Args; i++) {
        kernel.runtime_arg_schema.common_runtime_arg_names.push_back("v" + std::to_string(i));
    }
    kernel.source = m2::KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    volatile tt_l1_ptr uint32_t* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::address));
    destination[0] = get_arg(args::v0);
}
)"};
    const m2::ProgramSpec spec{
        .name = "bench_per_device_metal2",
        .kernels = {kernel},
        .work_units = {m2::test_helpers::MakeMinimalWorkUnit("main", m2::NodeCoord{0, 0}, {kMetal2Kernel})},
    };
    auto workload = m2::MakeMeshWorkloadFromSpec(mesh_device(), spec);
    const MeshCoordinateRange range(mesh_device().shape());
    auto& program = workload.get_programs().at(range);
    m2::SetProgramRunArgs(program, metal2_args(true, true, 0));
    std::vector<m2::ProgramRunArgs> per_device_args;
    for (uint32_t index = 1; index <= range.shape().mesh_size(); index++) {
        per_device_args.push_back(metal2_args(false, true, 1000 * index));
    }
    auto set_per_device = [&] {
        uint32_t index = 0;
        for (const auto& coord : range) {
            m2::SetProgramRunArgsPerDevice(workload, range, coord, per_device_args[index++]);
        }
    };
    if (mode != 0) {
        set_per_device();
    }
    const auto update_address = metal2_args(true, false, 0);
    const auto update_values = metal2_args(false, true, 0);
    if (mode == 3) {
        run(state, workload, set_per_device);
    } else if (mode == 2) {
        run(state, workload, [&] {
            m2::UpdateProgramRunArgs(program, update_values, /*skip_validation=*/true);
            set_per_device();
        });
    } else {
        run(state, workload, [&] { m2::UpdateProgramRunArgs(program, update_address, /*skip_validation=*/true); });
    }
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
    ->ArgNames({"fields", "reset"})
    ->Args({0, 0})
    ->Args({30, 0})
    ->Args({30, 1})
    ->Args({30, 2})
    ->Iterations(4000)
    ->Repetitions(3)
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Metal2)
    ->Name("BM_PerDeviceRuntimeArgs/Metal2")
    ->ArgName("mode")
    ->Arg(0)
    ->Arg(1)
    ->Arg(2)
    ->Arg(3)
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
