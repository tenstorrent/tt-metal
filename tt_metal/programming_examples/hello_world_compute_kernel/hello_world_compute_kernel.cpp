// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include "hostdevcommon/common_values.hpp"
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/dispatch_core_common.hpp"
#include "tt-metalium/kernel_types.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    // Initialize Program and Device

    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    IDevice* device = CreateDevice(
        device_id, 1, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, DispatchCoreConfig{DispatchCoreType::ETH});
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    auto max_x = device->logical_grid_size().x - 1;
    auto max_y = device->logical_grid_size().y - 1;
    // CoreRange core_range_0{{0,0}, {max_x, max_y / 2}};
    // CoreRange core_range_1{{0, max_y / 2 + 1}, {max_x, max_y}};
    // std::cout << "Physical max x y = " << device->virtual_core_from_logical_core({max_x, max_y},
    // CoreType::WORKER).str() << "\n"; std::cout << "Physical min x y = " << device->virtual_core_from_logical_core({0,
    // 0}, CoreType::WORKER).str() << "\n"; Configure and Create Void Kernel

    auto core_range_0 = CoreCoord{0, 0};
    auto core_range_1 = CoreCoord{1, 1};
    auto core_range_2 = CoreCoord{2, 2};

    // std::cout << fmt::format("Dispatch to {} + {} cores\n", core_range_0.size(), core_range_1.size());

    std::vector<uint32_t> compute_kernel_args = {};
    KernelHandle void_compute_kernel_2 = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
        core_range_2,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .opt_level = KernelBuildOptLevel::O3});
    KernelHandle void_compute_kernel_0 = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
        core_range_0,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .opt_level = KernelBuildOptLevel::O3});

    KernelHandle void_compute_kernel_1 = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_world_compute_kernel/kernels/compute/void_compute_kernel.cpp",
        core_range_1,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .opt_level = KernelBuildOptLevel::O3});

    // Configure Program and Start Program Execution on Device

    // SetRuntimeArgs(program, void_compute_kernel_0, core_range_0, {0xdeadbeef});
    std::this_thread::sleep_for(std::chrono::seconds(1));
    SetCommonRuntimeArgs(program, void_compute_kernel_0, {0x10101010, 0xaaaaaaaa});
    SetCommonRuntimeArgs(program, void_compute_kernel_1, {0x20202020, 0xbbbbbbbb});
    SetCommonRuntimeArgs(program, void_compute_kernel_2, {0x30303030, 0xcccccccc});
    EnqueueProgram(cq, program, false);
    printf("Hello, Core {0, 0} on Device 0, I am sending you a compute kernel. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    CloseDevice(device);

    return 0;
}
