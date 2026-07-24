// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SMOKING-GUN microbenchmark for issues #50932 / #50772.
//
// Question: does the host cost of EnqueueMeshWorkload scale with the NUMBER OF
// PROGRAMS in the workload, independent of device work?
//
// The MoE-prefill `DispatchDeviceOperation` builds ONE ProgramDescriptor per mesh
// coordinate (32 programs on an 8x4 Blackhole mesh), because per-device fabric
// connections + baked coord scalars force a distinct program per device. A normal
// op is ONE program spanning the whole mesh. `FDMeshCommandQueue::enqueue_mesh_workload`
// loops per program (get_dispatch_cmds / update_program_dispatch_commands /
// write_program_cmds_to_subgrid), so an N-program op pays ~N x the per-program host
// cost every enqueue -- even on a warm cache.
//
// This benchmark isolates that: identical trivial device kernel (blank), same set of
// physical devices, ONLY the program-COUNT varies. If host-enqueue time is ~linear
// in program count, Dispatch's 32-program structure provably pays ~32x a 1-program op.
//
// Build:  part of the `distributed_unit_tests` target (add this file to
//         tests/tt_metal/distributed/CMakeLists.txt DISTRIBUTED_UNIT_TEST_SOURCES).
// Run  :  TT_METAL_HOME=$PWD ./build/test/tt_metal/distributed/distributed_unit_tests \
//             --gtest_filter='*ProgramCountEnqueueCost*'
//         Reports the host-enqueue FLOOR (min over many samples), per the #50772
//         methodology (shared host is noisy; min is the stable estimator).

#include <chrono>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshWorkload;

// A trivial single-core blank kernel. Device work is ~0, so any enqueue-time delta
// between workloads is pure host-side per-program cost.
Program make_blank_program() {
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreCoord{0, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    return program;
}

// Build a workload with `num_programs` distinct programs, each pinned to a distinct
// single mesh coordinate (mirrors Dispatch's one-descriptor-per-coord structure).
// `num_programs == 0` is the sentinel for the "uniform" baseline: ONE program spanning
// the entire mesh (mirrors a normal op).
MeshWorkload build_workload(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t num_programs) {
    MeshWorkload workload;
    if (num_programs == 0) {
        workload.add_program(MeshCoordinateRange(mesh_device->shape()), make_blank_program());
        return workload;
    }
    uint32_t added = 0;
    for (const auto& coord : MeshCoordinateRange(mesh_device->shape())) {
        if (added >= num_programs) {
            break;
        }
        workload.add_program(MeshCoordinateRange(coord, coord), make_blank_program());
        ++added;
    }
    return workload;
}

// Returns the FLOOR (min) host-enqueue time in microseconds over `samples` warm enqueues.
double measure_enqueue_floor_us(MeshCommandQueue& cq, MeshWorkload& workload, int samples) {
    // Warm: first enqueue builds + caches the per-program dispatch command sequences.
    EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    Finish(cq);

    double best_us = 1e18;
    for (int i = 0; i < samples; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        auto t1 = std::chrono::steady_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        best_us = std::min(best_us, us);
        // Drain periodically so we measure host-issue time, not command-queue backpressure.
        if ((i & 0x7) == 0x7) {
            Finish(cq);
        }
    }
    Finish(cq);
    return best_us;
}

TEST_F(GenericMeshDeviceFixture, ProgramCountEnqueueCost) {
    auto& cq = mesh_device_->mesh_command_queue();
    const uint32_t num_devices = mesh_device_->num_devices();
    constexpr int kSamples = 2000;

    // Sweep: 0 = uniform baseline (1 program over full mesh), then per-coord counts.
    std::vector<uint32_t> sweep = {0, 1, 2, 4, 8, 16, num_devices};

    fmt::print("\n==== ProgramCountEnqueueCost (mesh = {} devices, floor of {} samples) ====\n", num_devices, kSamples);
    fmt::print("{:>28} | {:>16} | {:>14}\n", "workload", "host enqueue us", "us / program");

    double uniform_us = 0.0;
    for (uint32_t k : sweep) {
        if (k > num_devices) {
            continue;
        }
        MeshWorkload workload = build_workload(mesh_device_, k);
        double us = measure_enqueue_floor_us(cq, workload, kSamples);
        if (k == 0) {
            uniform_us = us;
            fmt::print("{:>28} | {:>16.2f} | {:>14}\n", "1 program (full-mesh range)", us, "-");
        } else {
            fmt::print(
                "{:>28} | {:>16.2f} | {:>14.2f}\n",
                fmt::format("{} programs (1 coord each)", k),
                us,
                us / static_cast<double>(k));
        }
    }
    fmt::print(
        "\nInterpretation: if host-enqueue us grows ~linearly in program count, the\n"
        "per-program slope is the avoidable cost. Dispatch runs at k = {} while a\n"
        "uniform op runs at the '1 program' row (~{:.1f} us).\n\n",
        num_devices,
        uniform_us);

    // Not an assertion on absolute time (host is shared/noisy); this benchmark exists to
    // emit the table. Sanity only: the sweep ran.
    SUCCEED();
}

}  // namespace
}  // namespace tt::tt_metal::distributed::test
