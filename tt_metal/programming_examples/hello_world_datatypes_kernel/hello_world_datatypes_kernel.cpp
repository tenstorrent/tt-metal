// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // This example demonstrates that though the RISC-V cores (Both the data movement and compute cores) are
    // RV32IM and the Tensix relies on the SFPU and FPU attached to the compute cores to perform the bulk of the
    // floating point operations, it is still possible to operate on floating point data types directly on the
    // RISC-V cores as they are fully programmable and the compiler can generate the necessary software floating point
    // operations.

    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // Initialize Program and Device
    constexpr CoreCoord core = {0, 0};
    IDevice* device = CreateDevice(0);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Define and Create Buffer with Float Data Type
    constexpr uint32_t buffer_size = 2 * 1024;
    tt_metal::BufferConfig buffer_config = {
        .device = device, .size = buffer_size, .page_size = buffer_size, .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> dram_buffer = CreateBuffer(buffer_config);

    // Configure and Create Circular Buffer (to move data from DRAM to L1)
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(buffer_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, buffer_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    // Configure and create the kernel
    KernelHandle data_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "hello_world_datatypes_kernel/kernels/dataflow/float_dataflow_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Initialize Float Data for the DRAM Buffer
    float init_data = 1.23;
    EnqueueWriteBuffer(cq, dram_buffer, &init_data, false);

    // Configure Program and Start Program Execution on Device
    SetRuntimeArgs(program, data_reader_kernel_id, core, {dram_buffer->address()});
    EnqueueProgram(cq, program, false);

    fmt::print("Hello, Core {{0, 0}} on Device 0, please handle the data.\n");

    // Wait Until Program Finishes, Print "Hello World!", and Close Device
    Finish(cq);
    fmt::print("Thank you, Core {{0, 0}} on Device 0, for handling the data.\n");
    CloseDevice(device);

    return 0;
}
