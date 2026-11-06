// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt-metalium/kernel_types.hpp"
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

using namespace tt;
using namespace tt::tt_metal;

// A bit of a hack to handle packaged examples but also work inside the Metalium git repo.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // Initialize mesh device, command queue, workload, and device range
    constexpr experimental::NodeCoord core = {0, 0};
    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(device_id);  // 1x1 mesh
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Configure and create the kernel
    // This kernel does not perform any computation, it simply prints a message to show the compute cores running.
    // Within a Tensix, 3 RISC-V cores run collaboratively to perform compute tasks. These are the UNPACK, MATH,
    // and PACK cores.
    // Please view the respective documentation for more information:
    // https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md#tenstorrent-architecture-overview
    const experimental::KernelSpecName void_compute_kernel{"void_compute_kernel"};
    experimental::KernelSpec void_compute_kernel_spec{
        .unique_id = void_compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    // A program spec describes the program: its kernels and the cores the kernels run on (the work units).
    experimental::ProgramSpec program_spec{
        .name = "hello_world_compute_kernel",
        .kernels = {void_compute_kernel_spec},
        .work_units = {{.name = "hello_world_compute_kernel", .kernels = {void_compute_kernel}, .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Configure program and enqueue it as a mesh workload (non-blocking)
    experimental::ProgramRunArgs run_args{.kernel_run_args = {{.kernel = void_compute_kernel}}};
    experimental::SetProgramRunArgs(program, run_args);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    fmt::print("Hello, Core (0, 0) on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    // Wait Until MeshWorkload Finishes. The kernel will print the following messages:
    // 0:(x=0,y=0):TR0: Hello, I am the UNPACK core running the compute kernel
    // 0:(x=0,y=0):TR1: Hello, I am the MATH core running the compute kernel
    // 0:(x=0,y=0):TR2: Hello, I am the PACK core running the compute kernel
    // Deconstructing the output:
    // 0: - Device ID
    // (x=0,y=0): - Tensix core coordinates (so the 3 compute cores are all on the same core {0, 0})
    // TR0: Compute core 0 (UNPACK)
    // TR1: Compute core 1 (MATH)
    // TR2: Compute core 2 (PACK)

    // Wait for the workload to finish execution, then close the mesh device.
    distributed::Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    mesh_device->close();
    return 0;
}
