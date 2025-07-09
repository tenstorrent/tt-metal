// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

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

    // Initialize Program and Device. We are going to use the first device (0) and the first core (0, 0) on the device.
    constexpr CoreCoord core = {0, 0};
    IDevice* device = CreateDevice(0);
    // Command queue lets us submit work (execute programs and read/write buffers) to the device.
    CommandQueue& cq = device->command_queue();
    // And a program that represents the set of work we want to execute on the device at a single time.
    Program program = CreateProgram();

    // Configure and create Data Movement kernels
    // There are 2 Data Movement cores per Tensix. In applications one is usually used for reading data from DRAM and
    // the other for writing data. However for demonstration purposes, we will create 2 Data Movement kernels that
    // simply prints a message to show them running at the same time.
    KernelHandle data_movement_kernel_0 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle data_movement_kernel_1 = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Set Runtime Arguments for the Data Movement Kernels (none in this case and execute the program)
    SetRuntimeArgs(program, data_movement_kernel_0, core, {});
    SetRuntimeArgs(program, data_movement_kernel_1, core, {});
    std::cout << "Hello, Core {0, 0} on Device 0, Please start execution. I will standby for your communication."
              << std::endl;

    EnqueueProgram(cq, program, false);
    Finish(cq);
    // Wait Until Program Finishes. The program should print the following (NC and BR is Data movement core 1 and 0
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
    CloseDevice(device);
    return 0;
}
