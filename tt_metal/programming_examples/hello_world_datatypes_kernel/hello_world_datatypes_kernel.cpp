// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    // Initialize Program and Device

    constexpr CoreCoord core = {0, 0};
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Define and Create Buffer with Float Data Type

    constexpr uint32_t buffer_size = 2 * 1024;
    tt_metal::BufferConfig buffer_config = {
        .device = device,
        .size = buffer_size,
        .page_size = buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config);

    // Configure and Create Circular Buffer (to move data from DRAM to L1)

    constexpr uint32_t src0_cb_index = CB::c_in0;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(buffer_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, buffer_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Configure and Create Data Movement Kernels

    KernelHandle data_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_world_datatypes_kernel/kernels/dataflow/float_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Initialize Float Data for the DRAM Buffer

    float init_data = 1.23;
    EnqueueWriteBuffer(cq, dram_buffer, &init_data, false);

    // Configure Program and Start Program Execution on Device

    SetRuntimeArgs(program, data_reader_kernel_id, core, {dram_buffer->address()});
    EnqueueProgram(cq, program, false);

    printf("Hello, Core {0, 0} on Device 0, please handle the data.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    printf("Thank you, Core {0, 0} on Device 0, for handling the data.\n");
    CloseDevice(device);

    return 0;
}
