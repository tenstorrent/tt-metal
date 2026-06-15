// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase-0 baseline host program for the "kernel arguments as function & template parameters"
// work (see tech_reports/NamedKernelArgs/kernel_args_as_parameters.md).
//
// Built entirely on the EXISTING Metal 2.0 named-argument host API on main
// (experimental::KernelSpec / MakeProgramFromSpec / SetProgramRunArgs). It registers the §2
// worked-example schema — 3 named CTAs, 3 named RTAs, 1 named CRTA — sets their values by
// name, and runs the kernel, which DPRINTs every value. This is the fixture we incrementally
// port toward the Phase 1 design (TT_KERNEL marker + function/template parameters + a
// generated kernel_main() shim).
//
// Run with DPRINT enabled to see the kernel's output:
//   export TT_METAL_DPRINT_CORES=0,0
//   ./metal_example_named_kernel_args

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    if (std::getenv("TT_METAL_DPRINT_CORES") == nullptr) {
        std::cerr << "WARNING: set TT_METAL_DPRINT_CORES=0,0 to see the kernel's DPRINT output.\n"
                  << "         e.g. export TT_METAL_DPRINT_CORES=0,0\n";
    }

    // 1x1 mesh on the first device; the kernel runs on node/core (0, 0).
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(mesh_device->shape());

    const experimental::KernelSpecName KERNEL{"named_kernel_args"};
    const experimental::NodeCoord node{0, 0};

    // §2 worked-example schema: 3 CTAs, 3 RTAs, 1 CRTA. uint32_t-only (Phase 1).
    experimental::KernelSpec kernel_spec{
        .unique_id = KERNEL,
        .source = OVERRIDE_KERNEL_PREFIX "named_kernel_args/kernels/dataflow/named_kernel_args_kernel.cpp",
        .num_threads = 1,
        .compile_time_args = {{"Ht", 4}, {"Wt", 2}, {"untilize", 1}},  // 3 CTAs
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_tile_id", "num_tiles", "start_row"},  // 3 RTAs
                .common_runtime_arg_names = {"scaler"},                            // 1 CRTA
            },
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .gen1_config =
                    experimental::DataMovementHardwareConfig::Gen1Config{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
                .gen2_config = experimental::DataMovementHardwareConfig::Gen2Config{}},
    };

    experimental::ProgramSpec spec{
        .name = "named_kernel_args",
        .kernels = {kernel_spec},
        .work_units = {{.name = "wu", .kernels = {KERNEL}, .target_nodes = node}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    // Set the runtime-arg values by name. CTAs were already baked in at MakeProgramFromSpec.
    experimental::ProgramRunArgs params;
    params.kernel_run_args = {
        experimental::ProgramRunArgs::KernelRunArgs{
            .kernel = KERNEL,
            .runtime_arg_values = {{node, {{"start_tile_id", 64}, {"num_tiles", 8}, {"start_row", 3}}}},
            .common_runtime_arg_values = {{"scaler", 2}},
        },
    };
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);

    mesh_device->close();
    return 0;
}
