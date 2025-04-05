// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>

// Stand-alone example demonstrating usage of native multi-device TT-Metalium APIs
// for issuing a program dispatch across a mesh of devices.
int main() {
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::distributed;

    auto mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(2, 4)));
    auto& cq = mesh_device->mesh_command_queue();

    // In a typical single-device fashion, instantiate a program with
    // an example compute kernel targeting a 2x2 core range.
    auto example_program = CreateProgram();
    auto target_tensix_cores = CoreRange{
        CoreCoord{0, 0} /* start_coord */, CoreCoord{1, 1} /* end_coord */
    };

    auto compute_kernel_id = CreateKernel(
        example_program,
        "tt_metal/programming_examples/distributed/1_distributed_program_dispatch/kernels/void_kernel.cpp",
        target_tensix_cores,
        ComputeConfig{.compile_args = {}});

    // Configure the runtime arguments for the kernel.
    auto runtime_args = std::vector<uint32_t>{};
    SetRuntimeArgs(example_program, compute_kernel_id, target_tensix_cores, runtime_args);

    // Instantiate a MeshWorkload and attach the example program. We'll broadcast
    // this program by enqueueing it across all devices in our 2x4 mesh.
    auto mesh_workload = CreateMeshWorkload();
    auto target_devices = MeshCoordinateRange(mesh_device->shape());

    AddProgramToMeshWorkload(mesh_workload, std::move(example_program), target_devices);
    EnqueueMeshWorkload(cq, mesh_workload, false /* blocking */);

    // Synchronize the mesh command queue to ensure the workload has completed.
    Finish(cq);

    return 0;
}
