// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include "tt-metalium/kernel_types.hpp"

using namespace tt;
using namespace tt::tt_metal;

// A bit of a hack to handle packaged examples but also work inside the Metalium git repo.
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of "
                     "the Data Movement kernels."
                  << std::endl;
        std::cerr << "WARNING: For example, export TT_METAL_DPRINT_CORES=0,0" << std::endl;
    }

    // Initialize Program and Device
    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Configure and create the kernel
    // This kernel does not perform any computation, it simply prints a message to show the compute cores running.
    // Within a Tensix, 3 RISC-V cores run collaboratively to perform compute tasks. These are the UNPACK, MATH,
    // and PACK cores.
    // Please view the respective documentation for more information:
    // https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md#tenstorrent-architecture-overview
    KernelHandle void_compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
        core,
        ComputeConfig{});

    // Configure Program and Start Program Execution on Device
    SetRuntimeArgs(program, void_compute_kernel_id, core, {});
    EnqueueProgram(cq, program, false);
    printf("Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    // Wait Until Program Finishes. The kernel will print the following messages:
    // 0:(x=0,y=0):TR0: Hello, I am the UNPACK core running the compute kernel
    // 0:(x=0,y=0):TR1: Hello, I am the MATH core running the compute kernel
    // 0:(x=0,y=0):TR2: Hello, I am the PACK core running the compute kernel
    // Deconstructing the output:
    // 0: - Device ID
    // (x=0,y=0): - Tensix core coordinates (so the 3 compute cores are all on the same core {0, 0})
    // TR0: Compute core 0 (UNPACK)
    // TR1: Compute core 1 (MATH)
    // TR2: Compute core 2 (PACK)

    // Wait for the program to finish execution.
    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    CloseDevice(device);
    return 0;
}
