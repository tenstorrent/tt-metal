// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/command_list.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "tests/tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"

namespace tt::tt_metal::experimental::bench {
namespace {

namespace m2 = tt::tt_metal::experimental;
using namespace tt::tt_metal::distributed;
using m2::test_helpers::BindTensorParameterToKernel;
using m2::test_helpers::MakeMinimalGen1DMKernel;
using m2::test_helpers::MakeMinimalWorkUnit;

// Operations per timed batch. Reported times are per operation.
constexpr uint32_t kBatch = 100;
constexpr uint32_t kIterations = 20;
constexpr uint32_t kNumNamedCrtas = 64;
constexpr m2::NodeCoord kNode{0, 0};
constexpr char kKernelName[] = "writer";

MeshDevice& get_device() {
    static std::shared_ptr<MeshDevice> mesh_device =
        MeshDevice::create(MeshDeviceConfig(MeshShape{1, 1}), DEFAULT_L1_SMALL_SIZE, 64 << 20);
    return *mesh_device;
}

bool is_supported(benchmark::State& state, MeshDevice& mesh_device) {
    const auto arch = mesh_device.get_devices().at(0)->arch();
    if (arch != tt::ARCH::WORMHOLE_B0 && arch != tt::ARCH::BLACKHOLE) {
        state.SkipWithError("Command lists require Wormhole B0 or Blackhole hardware");
        return false;
    }
    return true;
}

// A built command list and two patches with different values, so alternating them changes every parameter.
struct BenchList {
    // The patches reference these tensors.
    std::vector<MeshTensor> tensors;
    CommandList list;
    std::array<CmdListArgPatch, 2> patches;
};

m2::ProgramSpec make_spec(uint32_t index, bool tensor) {
    auto kernel = MakeMinimalGen1DMKernel(kKernelName, DataMovementProcessor::RISCV_0);
    kernel.runtime_arg_schema.runtime_arg_names = {"address"};
    m2::ProgramSpec spec{
        .name = (tensor ? "bench_tensor_" : "bench_scalar_") + std::to_string(index),
        .work_units = {MakeMinimalWorkUnit("main", kNode, {kKernelName})},
    };
    if (tensor) {
        kernel.source = m2::KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    TensorAccessor accessor(tensor::io);
    volatile tt_l1_ptr uint32_t* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::address));
    destination[0] = accessor.get_bank_base_address();
}
)"};
        BindTensorParameterToKernel(kernel, "io", "io");
    } else {
        kernel.source = m2::KernelSpec::SourceCode{R"(
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
void kernel_main() {
    volatile tt_l1_ptr uint32_t* destination = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::address));
    destination[0] = get_arg(args::v0);
}
)"};
        for (uint32_t i = 0; i < kNumNamedCrtas; ++i) {
            kernel.runtime_arg_schema.common_runtime_arg_names.push_back("v" + std::to_string(i));
        }
    }
    spec.kernels = {kernel};
    return spec;
}

// Records num_programs single-program workloads and registers num_params parameters.
// Scalar parameter i is CRTA v(i / num_programs) of program i % num_programs, so few programs pack the parameters
// into shared patch windows and many programs spread them out. Tensor parameter i is the tensor of program i.
BenchList make_list(MeshDevice& mesh_device, bool tensor, uint32_t num_programs, uint32_t num_params) {
    const auto address = static_cast<uint32_t>(mesh_device.allocator()->get_base_allocator_addr(HalMemType::L1));
    const auto tensor_spec = TensorSpec(
        Shape{2, 512},
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    std::vector<MeshTensor> tensors;
    if (tensor) {
        tensors.reserve(2);
        tensors.push_back(MeshTensor::allocate_on_device(mesh_device, tensor_spec));
        tensors.push_back(MeshTensor::allocate_on_device(mesh_device, tensor_spec));
    }

    std::array<CmdListArgPatch, 2> patches;
    CommandListBuilder builder(mesh_device);
    for (uint32_t p = 0; p < num_programs; ++p) {
        auto spec = make_spec(p, tensor);
        if (tensor) {
            spec.tensor_parameters = {m2::TensorParameter{.unique_id = m2::TensorParamName{"io"}, .spec = tensor_spec}};
        }
        auto workload = m2::MakeMeshWorkloadFromSpec(mesh_device, spec);
        auto& program = workload.get_programs().begin()->second;

        m2::ProgramRunArgs::KernelRunArgs kernel_args{
            .kernel = m2::KernelSpecName{kKernelName},
            .runtime_arg_values = m2::MakeRuntimeArgsForSingleNode(kNode, {{"address", address}}),
        };
        m2::ProgramRunArgs args;
        if (tensor) {
            args.tensor_args = {{m2::TensorParamName{"io"}, m2::ProgramRunArgs::TensorArgument{tensors[0]}}};
        } else {
            for (uint32_t i = 0; i < kNumNamedCrtas; ++i) {
                kernel_args.common_runtime_arg_values["v" + std::to_string(i)] = 0;
            }
        }
        args.kernel_run_args = {kernel_args};
        m2::SetProgramRunArgs(program, args);

        CmdListParameters parameters;
        if (tensor) {
            if (p < num_params) {
                const CmdListTensorArgName name{"tensor" + std::to_string(p)};
                parameters.tensor_parameters.emplace(
                    name,
                    std::vector<CmdListTensorArgInfo>{
                        {.program = std::cref(program), .param_name = m2::TensorParamName{"io"}}});
                patches[0].tensor_args.emplace(name, m2::ProgramRunArgs::TensorArgument{tensors[1]});
                patches[1].tensor_args.emplace(name, m2::ProgramRunArgs::TensorArgument{tensors[0]});
            }
        } else {
            for (uint32_t i = p; i < num_params; i += num_programs) {
                const CmdListCommonRuntimeArgName name{"param" + std::to_string(i)};
                parameters.common_runtime_parameters.emplace(
                    name,
                    std::vector<CmdListCommonRuntimeArgInfo>{
                        {.program = std::cref(program),
                         .kernel_name = m2::KernelSpecName{kKernelName},
                         .arg_name = "v" + std::to_string(i / num_programs)}});
                patches[0].common_runtime_args.emplace(name, i + 1);
                patches[1].common_runtime_args.emplace(name, i + 0x10001);
            }
        }
        builder.add(workload, parameters);
    }
    return BenchList{std::move(tensors), builder.build(mesh_device.mesh_command_queue(0)), std::move(patches)};
}

// run_batch runs kBatch operations and returns the seconds to report for them.
void run_timed(benchmark::State& state, const std::function<double()>& run_batch) {
    run_batch();
    double total_us = 0;
    for ([[maybe_unused]] auto _ : state) {
        const double seconds_per_op = run_batch() / kBatch;
        state.SetIterationTime(seconds_per_op);
        total_us += seconds_per_op * 1e6;
    }
    state.counters["us_per_op"] = total_us / static_cast<double>(state.iterations());
}

void set_shape_counters(benchmark::State& state, bool tensor, uint32_t num_programs, uint32_t num_params) {
    state.counters["kind"] = tensor ? 1 : 0;
    state.counters["programs"] = num_programs;
    state.counters["patched_params"] = num_params;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void BM_Replay(benchmark::State& state) {
    auto& mesh_device = get_device();
    if (!is_supported(state, mesh_device)) {
        return;
    }
    const auto num_programs = static_cast<uint32_t>(state.range(0));
    auto bench = make_list(mesh_device, /*tensor=*/false, num_programs, /*num_params=*/0);
    auto& cq = mesh_device.mesh_command_queue(0);
    run_timed(state, [&] {
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < kBatch; ++i) {
            EnqueueCommandList(cq, bench.list, /*blocking=*/false);
        }
        Finish(cq);
        return seconds_since(start);
    });
    set_shape_counters(state, false, num_programs, 0);
}

void BM_PatchAndReplay(benchmark::State& state) {
    auto& mesh_device = get_device();
    if (!is_supported(state, mesh_device)) {
        return;
    }
    const bool tensor = state.range(0) != 0;
    const auto num_programs = static_cast<uint32_t>(state.range(1));
    const auto num_params = static_cast<uint32_t>(state.range(2));
    auto bench = make_list(mesh_device, tensor, num_programs, num_params);
    auto& cq = mesh_device.mesh_command_queue(0);
    run_timed(state, [&] {
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < kBatch; ++i) {
            bench.list.update_args(bench.patches[i % 2]);
            EnqueueCommandList(cq, bench.list, /*blocking=*/false);
        }
        Finish(cq);
        return seconds_since(start);
    });
    set_shape_counters(state, tensor, num_programs, num_params);
}

// Times only the host side of non-blocking update_args; the device work is drained outside the timed region.
void BM_UpdateArgsHost(benchmark::State& state) {
    auto& mesh_device = get_device();
    if (!is_supported(state, mesh_device)) {
        return;
    }
    const bool tensor = state.range(0) != 0;
    const auto num_programs = static_cast<uint32_t>(state.range(1));
    const auto num_params = static_cast<uint32_t>(state.range(2));
    auto bench = make_list(mesh_device, tensor, num_programs, num_params);
    auto& cq = mesh_device.mesh_command_queue(0);
    run_timed(state, [&] {
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < kBatch; ++i) {
            bench.list.update_args(bench.patches[i % 2]);
        }
        const double seconds = seconds_since(start);
        Finish(cq);
        return seconds;
    });
    set_shape_counters(state, tensor, num_programs, num_params);
}

void PatchSweep(benchmark::internal::Benchmark* b) {
    for (const int64_t num_programs : {1, 16, 128}) {
        for (const int64_t num_params : {1, 8, 64}) {
            b->Args({0, num_programs, num_params});
        }
    }
    for (const int64_t num_programs : {1, 16, 128}) {
        b->Args({1, num_programs, 1});
        if (num_programs > 1) {
            b->Args({1, num_programs, num_programs});
        }
    }
}

BENCHMARK(BM_Replay)
    ->ArgName("programs")
    ->Arg(1)
    ->Arg(16)
    ->Arg(128)
    ->Iterations(kIterations)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_PatchAndReplay)
    ->Apply(PatchSweep)
    ->ArgNames({"tensor", "programs", "params"})
    ->Iterations(kIterations)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

BENCHMARK(BM_UpdateArgsHost)
    ->Apply(PatchSweep)
    ->ArgNames({"tensor", "programs", "params"})
    ->Iterations(kIterations)
    ->UseManualTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace tt::tt_metal::experimental::bench
