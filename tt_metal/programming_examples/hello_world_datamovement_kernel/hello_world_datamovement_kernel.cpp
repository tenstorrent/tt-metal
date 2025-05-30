// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "hostdevcommon/common_values.hpp"
#include "tt-metalium/dispatch_core_common.hpp"
#include "tt-metalium/fabric_types.hpp"

int main() {
    using namespace tt;
    using namespace tt::tt_metal;

    // Initialize Program and Device
    tt::tt_metal::detail::InitializeFabricConfig(FabricConfig::FABRIC_1D);

    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    auto devices = tt::tt_metal::detail::CreateDevices(
        {0, 1},
        2,
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        tt_metal::DispatchCoreConfig{DispatchCoreType::ETH});
    IDevice* device = devices[0];
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Configure and Create Void DataMovement Kernels

    KernelHandle void_dataflow_kernel_noc0_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle void_dataflow_kernel_noc1_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_world_datamovement_kernel/kernels/dataflow/void_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Configure Program and Start Program Execution on Device

    SetRuntimeArgs(program, void_dataflow_kernel_noc0_id, core, {});
    SetRuntimeArgs(program, void_dataflow_kernel_noc1_id, core, {});
    EnqueueProgram(cq, program, false);
    printf("Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    tt::tt_metal::detail::CloseDevices(devices);

    return 0;
}
