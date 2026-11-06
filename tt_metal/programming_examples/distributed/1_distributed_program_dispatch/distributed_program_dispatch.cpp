// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

// Stand-alone example demonstrating usage of native multi-device TT-Metalium APIs
// for issuing a program dispatch across a mesh of devices.
int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    auto& cq = mesh_device->mesh_command_queue();

    // In a typical single-device fashion, describe a program with
    // an example compute kernel targeting a 2x2 core range.
    auto target_tensix_cores = experimental::NodeRange{
        experimental::NodeCoord{0, 0} /* start_coord */, experimental::NodeCoord{1, 1} /* end_coord */
    };

    const experimental::KernelSpecName compute_kernel{"void_compute"};
    experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = "tt_metal/programming_examples/distributed/1_distributed_program_dispatch/kernels/void_kernel.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .hw_config = experimental::ComputeHardwareConfig{},
    };

    experimental::ProgramSpec example_program_spec{
        .name = "distributed_program_dispatch",
        .kernels = {compute_kernel_spec},
        .work_units = {{.name = "void_compute", .kernels = {compute_kernel}, .target_nodes = target_tensix_cores}},
    };
    auto example_program = experimental::MakeProgramFromSpec(*mesh_device, example_program_spec);

    // Configure the runtime arguments for the kernel.
    auto runtime_args = experimental::ProgramRunArgs{};
    experimental::SetProgramRunArgs(example_program, runtime_args);

    // Instantiate a MeshWorkload and attach the example program. We'll broadcast
    // this program by enqueueing it across all devices in our 2x4 mesh.
    auto mesh_workload = MeshWorkload();
    auto target_devices = MeshCoordinateRange(mesh_device->shape());

    mesh_workload.add_program(target_devices, std::move(example_program));
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    // Synchronize the mesh command queue to ensure the workload has completed.
    Finish(cq);

    return 0;
}
