// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Which mechanism causes the eager op2op gap in the deepseek MoE prefill Dispatch op
// (#50932 / #50772)?
//
// Already ruled out by measurement: host-side command generation (HOST DURATION 165 us,
// cross-validated by a microbenchmark at 161 us), the worker kernels themselves, and the
// cross-device barriers inside the kernel (~301 us of a ~1.36 ms kernel).
//
// Two candidates remain for the ~2.7 ms of dead time between kernels:
//   (2) REFILL-LIMITED: eager mode cannot run far enough ahead. The host issues op N+1 close
//       behind op N, the chain is serially dependent, and the command feed has latency, so the
//       dispatcher runs dry between ops. This is what trace deletes by pre-staging the tape.
//   (3) DISPATCHER-LIMITED: the on-chip command processor is genuinely busy -- for a 32-program
//       workload it parses 32 programs' commands and fires 32 go-signals.
//
// These respond OPPOSITELY to queue depth, which is the discriminator:
//   * refill-limited     -> per-workload time FALLS as depth rises (staging ahead helps)
//   * dispatcher-limited -> per-workload time is FLAT in depth, but RISES with program count K
//
// Measures steady-state throughput (enqueue `depth` workloads, then one Finish), NOT single-op
// host latency -- so it captures device-side dispatch cost that host-side timing cannot see.
// Deliberately runs with NO profiler: no device zones, no --sync-host-device, nothing that
// perturbs pipelining. That also makes it immune to the instrument inflation that makes the
// Tracy-measured 3471 us an upper bound.
//
// Run:
//   ./build/test/tt_metal/distributed/distributed_unit_tests \
//     --gtest_filter='*DispatchStarvation*'

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/core_coord.hpp>

#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace tt::tt_metal::distributed::test {
namespace {

using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshCoordinateRange;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshWorkload;

constexpr uint32_t kWarmup = 20;
constexpr uint32_t kSamples = 200;

// Blank kernel: no CBs, no runtime args, one core. Device work is ~nil, so anything we measure
// beyond host enqueue is dispatch-path latency rather than compute.
Program make_blank_program() {
    Program program = CreateProgram();
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        CoreCoord{0, 0},
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    return program;
}

// `num_programs` == 1 -> a single program spanning the whole mesh (what a uniform op does).
// `num_programs` > 1 -> one program per coordinate (what Dispatch does), same total devices.
MeshWorkload build_workload(const std::shared_ptr<MeshDevice>& mesh_device, uint32_t num_programs) {
    MeshWorkload workload;
    const auto shape = mesh_device->shape();
    if (num_programs <= 1) {
        workload.add_program(MeshCoordinateRange(shape), make_blank_program());
        return workload;
    }
    uint32_t added = 0;
    for (const auto& coord : MeshCoordinateRange(shape)) {
        if (added >= num_programs) {
            break;
        }
        workload.add_program(MeshCoordinateRange(coord, coord), make_blank_program());
        ++added;
    }
    return workload;
}

}  // namespace

TEST_F(GenericMeshDeviceFixture, DispatchStarvationQueueDepth) {
    auto& cq = mesh_device_->mesh_command_queue();
    const uint32_t num_devices = mesh_device_->num_devices();

    const std::vector<uint32_t> program_counts = {1, num_devices};
    const std::vector<uint32_t> depths = {1, 2, 4, 8, 16};

    fmt::print("\n==== DispatchStarvationQueueDepth (mesh = {} devices, floor of {} samples) ====\n",
               num_devices, kSamples);
    fmt::print("Steady-state per-workload wall time: enqueue `depth` workloads, then one Finish.\n");
    fmt::print("No profiler, no sync-host-device -- pipelining is undisturbed.\n\n");
    fmt::print("{:>9} {:>7} {:>18} {:>16}\n", "programs", "depth", "us / workload", "vs depth=1");

    for (uint32_t k : program_counts) {
        double baseline = 0.0;
        for (uint32_t depth : depths) {
            // Distinct workloads so nothing is deduplicated; all identical in shape/cost.
            std::vector<MeshWorkload> workloads;
            workloads.reserve(depth);
            for (uint32_t d = 0; d < depth; ++d) {
                workloads.push_back(build_workload(mesh_device_, k));
            }

            for (uint32_t w = 0; w < kWarmup; ++w) {
                for (auto& wl : workloads) {
                    EnqueueMeshWorkload(cq, wl, /*blocking=*/false);
                }
                Finish(cq);
            }

            double best_us_per_workload = std::numeric_limits<double>::max();
            for (uint32_t s = 0; s < kSamples; ++s) {
                auto t0 = std::chrono::steady_clock::now();
                for (auto& wl : workloads) {
                    EnqueueMeshWorkload(cq, wl, /*blocking=*/false);
                }
                Finish(cq);
                auto t1 = std::chrono::steady_clock::now();
                const double us =
                    std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(depth);
                best_us_per_workload = std::min(best_us_per_workload, us);
            }

            if (depth == depths.front()) {
                baseline = best_us_per_workload;
            }
            fmt::print("{:>9} {:>7} {:>18.2f} {:>15.2f}x\n",
                       k, depth, best_us_per_workload,
                       baseline > 0 ? best_us_per_workload / baseline : 1.0);
        }
        fmt::print("\n");
    }

    fmt::print("Read it as:\n");
    fmt::print("  per-workload time FALLS as depth rises  -> REFILL-LIMITED (candidate 2):\n");
    fmt::print("      the dispatcher was starved of input; staging ahead fixes it. Trace/issue-ahead\n");
    fmt::print("      is the lever, not the op.\n");
    fmt::print("  FLAT in depth but RISES with program count -> DISPATCHER-LIMITED (candidate 3):\n");
    fmt::print("      the on-chip command processor is the serial bottleneck. Device-side dispatch\n");
    fmt::print("      (32 programs -> 32 go-signals) is the lever.\n");
    fmt::print("  FLAT in both, near host-enqueue cost -> neither; the op is host-bound after all.\n");
}

}  // namespace tt::tt_metal::distributed::test
