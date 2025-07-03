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
    // Initialize Program and Device

    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Configure and Create Void Kernel

    std::vector<uint32_t> compute_kernel_args = {};
    KernelHandle void_compute_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .opt_level = KernelBuildOptLevel::O3});

    // Configure Program and Start Program Execution on Device

    SetRuntimeArgs(program, void_compute_kernel_id, core, {});
    EnqueueProgram(cq, program, false);
    printf("Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    CloseDevice(device);

    return 0;
}
