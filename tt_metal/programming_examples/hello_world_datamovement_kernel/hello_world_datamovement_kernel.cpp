// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include "tt-metalium/fabric_types.hpp"
#include "tt-metalium/tt_metal.hpp"

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    using namespace tt;
    using namespace tt::tt_metal;

    // Initialize Program and Device

    // detail::SetFabricConfig(FabricConfig::FABRIC_1D, FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);

    constexpr CoreCoord core = {0, 0};
    int device_id = 5;
    std::vector<int> devices_to_open;
    devices_to_open.reserve(GetNumAvailableDevices());
    for (int i = 0; i < GetNumAvailableDevices(); ++i) {
        devices_to_open.push_back(i);
    }
    auto devices = detail::CreateDevices(devices_to_open);
    Program program = CreateProgram();

    // Configure and Create Void DataMovement Kernels

    KernelHandle void_dataflow_kernel_noc0_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle void_dataflow_kernel_noc1_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Configure Program and Start Program Execution on Device

    SetRuntimeArgs(program, void_dataflow_kernel_noc0_id, core, {});
    SetRuntimeArgs(program, void_dataflow_kernel_noc1_id, core, {});
    // for (auto [device_id, device] : devices) {
    //     if (device_id == 0 || device_id == 1 || device_id == 2 || device_id == 3) {
    //         continue;
    //     }

    auto device = devices[17];
    std::cerr << "Begin Enqueeing programs on Device " << device_id << std::endl;
    CommandQueue& cq = device->command_queue();
    for (int i = 0; i < 100; ++i) {
        EnqueueProgram(cq, program, true);
    }
    Finish(cq);
    std::cerr << "Device " << device_id << " has finished enqueuing programs." << std::endl;
    // }
    printf("Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    detail::CloseDevices(devices);

    return 0;
}
