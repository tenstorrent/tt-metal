// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Experiment 2 for #50932/#50772: does the host cost of a 32-program mesh workload
// match Dispatch's ~3.5 ms once each program carries a REALISTIC footprint?
//
// The earlier program-COUNT benchmark (test_program_count_enqueue_cost.cpp) used a
// blank 1-core/0-arg kernel and found +99 us for 32 programs -> program count alone is
// cheap. But that under-tests Artem's theory: real Dispatch programs have
//   M = num_links senders + num_links*num_untilizers untilize cores, and
//   K = a large per-core runtime-arg vector (fabric routing-plane connection args).
// `update_program_dispatch_commands` (the per-program host loop in enqueue) scales with
// TOTAL runtime args = M*K, so a heavy program can cost far more per enqueue than a blank
// one. This benchmark sweeps (cores M, rt-args-per-core K) x {1 program, 32 programs} so
// the 32-vs-1 host delta is measured at Dispatch-like M*K, not at zero.
//
// CONFIRM/DENY: if at Dispatch-like (M,K) the (32 programs - 1 program) host delta
// approaches ~3.5 ms, Artem's "32 programs is the cost, collapse to 1" holds. If it stays
// in the hundreds of us, the 3.5 ms is not the host per-program loop.
//
// Build: add to tests/tt_metal/distributed/CMakeLists.txt DISTRIBUTED_UNIT_TEST_SOURCES.
// Run  : distributed_unit_tests --gtest_filter='*ProgramFootprintEnqueueCost*'

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshWorkload;

// One program: blank kernel on `num_cores` cores, each carrying `args_per_core` runtime
// args. The kernel ignores the args; only the host-side command volume (M*K) is under test,
// which is exactly what update_program_dispatch_commands processes per enqueue.
Program make_footprint_program(
    const std::shared_ptr<MeshDevice>& mesh_device, uint32_t num_cores, uint32_t args_per_core) {
    Program program = CreateProgram();
    auto grid = mesh_device->compute_with_storage_grid_size();
    CoreRangeSet cores = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);

    auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> rt_args(args_per_core, 0u);
    for (const auto& core : corerange_to_cores(cores)) {
        SetRuntimeArgs(program, kernel, core, rt_args);
    }
    return program;
}

MeshWorkload build_workload(
    const std::shared_ptr<MeshDevice>& mesh_device, uint32_t num_programs, uint32_t num_cores, uint32_t args_per_core) {
    MeshWorkload workload;
    if (num_programs == 1) {
        workload.add_program(
            MeshCoordinateRange(mesh_device->shape()), make_footprint_program(mesh_device, num_cores, args_per_core));
        return workload;
    }
    uint32_t added = 0;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (added >= num_programs) {
            break;
        }
        workload.add_program(
            MeshCoordinateRange(coord, coord), make_footprint_program(mesh_device, num_cores, args_per_core));
        ++added;
    }
    return workload;
}

double measure_enqueue_floor_us(MeshCommandQueue& cq, MeshWorkload& workload, int samples) {
    EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    Finish(cq);
    double best_us = 1e18;
    for (int i = 0; i < samples; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        auto t1 = std::chrono::steady_clock::now();
        best_us = std::min(best_us, std::chrono::duration<double, std::micro>(t1 - t0).count());
        if ((i & 0x7) == 0x7) {
            Finish(cq);
        }
    }
    Finish(cq);
    return best_us;
}

TEST_F(GenericMeshDeviceFixture, ProgramFootprintEnqueueCost) {
    auto& cq = mesh_device_->mesh_command_queue();
    const uint32_t nd = mesh_device_->num_devices();
    constexpr int kSamples = 2000;

    // (cores M, args/core K) grid bracketing Dispatch's real footprint.
    // Dispatch M = num_links*(1+num_untilizers) (small, ~4-24); K = fabric conn args (tens-hundreds).
    struct FP {
        uint32_t cores, args;
    };
    std::vector<FP> footprints = {{1, 0}, {8, 32}, {8, 128}, {24, 64}, {24, 256}, {64, 128}, {64, 512}};

    fmt::print("\n==== ProgramFootprintEnqueueCost (mesh={} devices, floor/{} samples) ====\n", nd, kSamples);
    fmt::print(
        "{:>10} {:>10} | {:>14} | {:>14} | {:>14}\n",
        "cores(M)",
        "args(K)",
        "1-prog us",
        "32-prog us",
        "32-vs-1 delta");
    for (const auto& fp : footprints) {
        MeshWorkload w1 = build_workload(mesh_device_, 1, fp.cores, fp.args);
        double us1 = measure_enqueue_floor_us(cq, w1, kSamples);
        MeshWorkload w32 = build_workload(mesh_device_, std::min<uint32_t>(32, nd), fp.cores, fp.args);
        double us32 = measure_enqueue_floor_us(cq, w32, kSamples);
        fmt::print("{:>10} {:>10} | {:>14.2f} | {:>14.2f} | {:>14.2f}\n", fp.cores, fp.args, us1, us32, us32 - us1);
    }
    fmt::print(
        "\nConfirm/deny: if the 32-vs-1 delta at Dispatch-like (M,K) approaches ~3500 us,\n"
        "the 32-program host loop IS the cost. If it stays in the hundreds, it is not.\n\n");
    SUCCEED();
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
