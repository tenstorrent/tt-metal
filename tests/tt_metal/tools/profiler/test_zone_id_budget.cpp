// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Unit test for KERNEL_PROFILER structural zone ids (id = (file_id << LOCAL_BITS) | local).
//
//   Case 1: a kernel with 120 profiler zones in one translation unit compiles and profiles -- the
//           per-TU zone budget (128 with LOCAL_BITS=7) has headroom. Because this run also builds the
//           per-RISC firmware (each its own TU with its own file_id), it exercises cross-TU id
//           uniqueness: the run reaching ReadMeshDeviceProfilerResults without the host-side
//           collision check firing proves the ids are collision-free across all those TUs.
//   Case 2: a kernel with >128 zones in one TU must FAIL to compile -- the static_assert in
//           TT_ZONE_META turns the overflow into a hard build error, which surfaces on the host as an
//           exception. The test passes only if that build throws.
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

using namespace tt;
using namespace tt::tt_metal;

// Build + enqueue a program whose data-movement kernels are kernels/zone_budget.cpp. When over_budget
// is set the kernel declares more zones than the per-TU budget allows, which must fail to compile.
void RunZoneBudgetKernel(const std::shared_ptr<distributed::MeshDevice>& mesh_device, bool over_budget) {
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());
    Program program = CreateProgram();

    std::map<std::string, std::string> defines;
    if (over_budget) {
        defines["ZONE_OVER_BUDGET"] = "1";
    }

    const std::string kernel = "tests/tt_metal/tools/profiler/kernels/zone_budget.cpp";
    const CoreCoord core = {0, 0};

    CreateKernel(
        program,
        kernel,
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines});
    if (!over_budget) {
        // A second processor gives a second kernel TU (a distinct file_id) for the cross-TU check.
        CreateKernel(
            program,
            kernel,
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .defines = defines});
    }

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

int main() {
    bool pass = true;

    try {
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

        // Case 1: 120 zones in one TU must compile and profile (and the whole run stays collision-free
        // across all the firmware/kernel TUs, else ReadMeshDeviceProfilerResults would throw).
        RunZoneBudgetKernel(mesh_device, /*over_budget=*/false);
        ReadMeshDeviceProfilerResults(*mesh_device);

        // Case 2: >128 zones in one TU must fail to build via the TT_ZONE_META static_assert.
        bool over_budget_threw = false;
        try {
            RunZoneBudgetKernel(mesh_device, /*over_budget=*/true);
        } catch (const std::exception& e) {
            over_budget_threw = true;
            const std::string what = e.what();
            // substr clamps when the string is shorter than the requested length.
            fmt::print("Over-budget kernel correctly failed to build: {}\n", what.substr(0, 300));
        }
        TT_FATAL(
            over_budget_threw,
            "Over-budget kernel (>128 zones in one TU) built successfully, but it must have tripped the "
            "KERNEL_PROFILER_LOCAL_BITS static_assert and failed to compile");

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
