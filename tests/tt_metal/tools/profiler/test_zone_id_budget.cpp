// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unit test for KERNEL_PROFILER structural zone ids (id = (file_id << LOCAL_BITS) | local).
//
//   Case 1: a kernel with 40 profiler zones in one translation unit compiles and profiles -- the
//           per-TU zone-id budget (64 with LOCAL_BITS=6) has headroom. Because this run also builds
//           the per-RISC firmware (each its own TU with its own file_id), it exercises cross-TU id
//           uniqueness: reaching ReadMeshDeviceProfilerResults without the host-side collision check
//           firing proves the ids are collision-free across all those TUs. The kernel runs on RISCV_0
//           (BRISC) because its code region is large (1.4 MB); NCRISC has only 16 KB of IRAM, a
//           code-SIZE limit that is unrelated to the zone-id budget; BRISC's large region fits it regardless.
//   Case 2: a kernel with more zones than the per-TU budget must FAIL to compile -- the static_assert in
//           TT_ZONE_META turns the id-budget overflow into a hard build error, which surfaces on the
//           host as an exception. The test passes only if that build throws with the static_assert.
//           The compiler/build errors it logs are INTENTIONAL (see the banner printed below).
//
// Requires profiling enabled (TT_METAL_DEVICE_PROFILER=1); otherwise the zone macros are no-ops and
// there is no static_assert to trip. The pytest harness sets it.

#include <cstdint>
#include <cstring>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>

#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;

// Build + enqueue a program whose kernel is kernels/zone_budget.cpp on RISCV_0 (BRISC). When
// over_budget is set the kernel declares more zones than the per-TU id budget allows, which must fail
// to compile (arch-independently, at the static_assert, before any code-size/link check).
void RunZoneBudgetKernel(const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool over_budget) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    std::map<std::string, std::string> defines;
    if (over_budget) {
        defines["ZONE_OVER_BUDGET"] = "1";
    }

    CreateKernel(
        program,
        "tests/tt_metal/tools/profiler/kernels/zone_budget.cpp",
        CoreCoord{0, 0},
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

int main() {
    bool pass = true;

    try {
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

        // The zone-budget kernel is built with -DZONE_OVER_BUDGET = 240 zones in one TU. Whether that is
        // "over budget" depends on the split: default LOCAL_BITS=6 allows 64/TU, more-zone-names mode
        // (rtoptions TT_METAL_PROFILER_MORE_ZONE_NAMES) LOCAL_BITS=9 allows 512/TU. The JIT reads the same
        // rtoptions flag, so the modes stay consistent.
        const bool more_zone_names = tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_more_zone_names();

        if (more_zone_names) {
            // 512-zone/TU budget: the 240-zone kernel that overflows the default 64-zone budget must now
            // COMPILE and profile. Reaching ReadMeshDeviceProfilerResults without throwing proves it.
            RunZoneBudgetKernel(mesh_device, /*over_budget=*/true);
            ReadMeshDeviceProfilerResults(*mesh_device);
            fmt::print("More-zone-names mode: 240-zone (>64) single-TU kernel compiled and profiled.\n");
        } else {
            // Case 1: 40 zones in one TU must compile and profile (and the whole run stays collision-free
            // across all the firmware/kernel TUs, else ReadMeshDeviceProfilerResults would throw).
            RunZoneBudgetKernel(mesh_device, /*over_budget=*/false);
            ReadMeshDeviceProfilerResults(*mesh_device);

            // Case 2: the 240-zone kernel must fail to build via the TT_ZONE_META static_assert. The
            // build errors logged next are EXPECTED -- the test verifies they happen and are the assert.
            fmt::print(
                "=== Building an over-budget kernel on purpose; the compiler/build errors that follow are "
                "EXPECTED and are caught below. ===\n");
            bool over_budget_threw = false;
            std::string what;
            try {
                RunZoneBudgetKernel(mesh_device, /*over_budget=*/true);
            } catch (const std::exception& e) {
                over_budget_threw = true;
                what = e.what();
            }
            TT_FATAL(
                over_budget_threw,
                "Over-budget kernel (more zones than the per-TU budget) built successfully, but it must "
                "have tripped the KERNEL_PROFILER_LOCAL_BITS static_assert and failed to compile");
            // Confirm it failed for the RIGHT reason (the id-budget static_assert), not an unrelated error.
            TT_FATAL(
                what.find("too many KERNEL_PROFILER zones") != std::string::npos,
                "Over-budget kernel failed to build, but not via the expected KERNEL_PROFILER_LOCAL_BITS "
                "static_assert. Error was: {}",
                what.substr(0, 500));
            fmt::print("Over-budget kernel correctly failed to build via the id-budget static_assert.\n");
        }

        pass &= mesh_device->close();

    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "{}\n", e.what());
        fmt::print(stderr, "System error message: {}\n", std::strerror(errno));
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
