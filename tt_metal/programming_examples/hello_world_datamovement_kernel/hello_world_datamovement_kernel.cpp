// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    using namespace tt;
    using namespace tt::tt_metal;

    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Data Movement kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // Initialize mesh device (1x1), command queue, workload, and device range.
    // We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr experimental::NodeCoord core = {0, 0};
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    // Prepare a workload and a device coordinate range that spans the mesh.
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Configure and create Data Movement kernels
    // There are 2 Data Movement cores per Tensix. In applications one is usually used for reading data from DRAM and
    // the other for writing data. However for demonstration purposes, we will create 2 Data Movement kernels that
    // simply prints a message to show them running at the same time.
    // Both kernel specs use the same kernel source; each one names its own Data Movement core and NoC.
    const experimental::KernelSpecName data_movement_kernel_0{"data_movement_kernel_0"};
    const experimental::KernelSpecName data_movement_kernel_1{"data_movement_kernel_1"};

    experimental::KernelSpec data_movement_kernel_0_spec{
        .unique_id = data_movement_kernel_0,
        .source = OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };

    experimental::KernelSpec data_movement_kernel_1_spec{
        .unique_id = data_movement_kernel_1,
        .source = OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    // A program spec describes the program: its kernels and the cores the kernels run on (the work units).
    experimental::ProgramSpec program_spec{
        .name = "hello_world_datamovement_kernel",
        .kernels = {data_movement_kernel_0_spec, data_movement_kernel_1_spec},
        .work_units =
            {{.name = "hello_world_datamovement_kernel",
              .kernels = {data_movement_kernel_0, data_movement_kernel_1},
              .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Set Runtime Arguments for the Data Movement Kernels (none in this case and execute the program)
    experimental::ProgramRunArgs run_args{
        .kernel_run_args = {{.kernel = data_movement_kernel_0}, {.kernel = data_movement_kernel_1}},
    };
    experimental::SetProgramRunArgs(program, run_args);
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
              << std::endl;

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    // Wait Until MeshWorkload Finishes. The program should print the following (NC and BR is Data movement core 1 and 0
    // respectively):
    //
    // 0:(x=0,y=0):NC: My logical coordinates are 0,0
    // 0:(x=0,y=0):NC: Hello, host, I am running a void data movement kernel on Data Movement core 1.
    // 0:(x=0,y=0):BR: My logical coordinates are 0,0
    // 0:(x=0,y=0):BR: Hello, host, I am running a void data movement kernel on Data Movement core 0.
    //
    // Deconstructing the output:
    // 0: - Device ID
    // (x=0,y=0): - Tensix core coordinates (so both Data Movement cores are on the same Tensix core
    // NC: - Data Movement core 1
    // BR: - Data Movement core 0

    std::cout << "Thank you, Core {0, 0} on Device 0, for the completed task." << std::endl;
    mesh_device->close();
    return 0;
}
