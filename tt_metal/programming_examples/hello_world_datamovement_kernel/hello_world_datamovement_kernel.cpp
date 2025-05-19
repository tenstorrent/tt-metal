// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include "tt-metalium/fabric_types.hpp"
#include <tt-metalium/event.hpp>

int main() {
    using namespace tt;
    using namespace tt::tt_metal;

    // Initialize Program and Device
    tt::tt_metal::detail::InitializeFabricConfig(tt::tt_metal::FabricConfig::FABRIC_1D);

    constexpr CoreCoord core = {0, 0};
    auto devices = tt::tt_metal::detail::CreateDevices({0, 1});
    IDevice* device = devices[1];
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

    constexpr uint32_t transfer_size = 64 * 1024;
    constexpr uint32_t page_size = transfer_size;
    constexpr tt_metal::BufferType buffer_type = tt_metal::BufferType::L1;

    auto buffer = CreateBuffer(InterleavedBufferConfig{device, transfer_size, page_size, buffer_type});
    auto future_event = std::make_shared<Event>();

    std::vector<uint32_t> src_vec(transfer_size / sizeof(uint32_t));
    for (int i = 0; i < transfer_size / sizeof(uint32_t); ++i) {
        src_vec[i] = i + 1;
    }
    std::vector<uint32_t> dst_vec;

    std::cout << "Generated write buffer" << std::endl;
    for (int i = 0; i < 25; ++i) {
        dst_vec.clear();
        std::cout << "EnqueueWriteBuffer " << i << std::endl;
        // EnqueueWriteBuffer(cq, buffer, src_vec, false);
        EnqueueProgram(cq, program, false);
        EnqueueReadBuffer(cq, buffer, dst_vec, true);
        // if (src_vec != dst_vec) {
        //     printf("Error: src_vec != dst_vec\n");
        //     return 1;
        // }
    }
    printf("Hello, Core {0, 0} on Device 0, I am sending you some data. Standby awaiting communication.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for the completed task.\n");
    tt::tt_metal::detail::CloseDevices(devices);

    return 0;
}
