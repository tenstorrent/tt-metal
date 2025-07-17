// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include "tt-metalium/buffer.hpp"

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    // Ensure printing from kernel is enabled (so we can see the output of the Data Movement kernels).
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        fmt::print(
            "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to 0,0 to see the output of the Data "
            "Movement kernels.\n");
        fmt::print("WARNING: For example, export TT_METAL_DPRINT_CORES=0,0\n");
    }

    // Initialize a device
    IDevice* device = CreateDevice(0);

    // Create a command queue and program
    // Command queue
    //    * Submit work (execute programs and read/write buffers) to the device
    // Program
    //    * Contains kernels that perform computations or data movement
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    // We will only be using one Tensix core for this particular example. As Tenstorrent processors are a 2D grid of
    // cores we can specify the core coordinates as (0, 0).
    constexpr CoreCoord core = {0, 0};

    // Adding 2 integers in RISC-V thus a buffer size of 4 bytes.
    constexpr uint32_t buffer_size = sizeof(uint32_t);
    // There are many modes of buffer allocation, here we use interleaved buffers. Interleaved buffers are the most
    // flexible and generally recommended buffer type for most applications. As the Tensix core does not have direct
    // access to DRAM, an extra buffer on L1 (SRAM) is required to read/write data from/to DRAM.
    // page_size is the size of each page in the buffer. In most applications this will be set to the size of a tile.
    // But for this example we set it to the size of a single integer as that is what we are adding.
    InterleavedBufferConfig dram_config{
        .device = device, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig l1_config{
        .device = device, .size = buffer_size, .page_size = buffer_size, .buffer_type = BufferType::L1};

    // Create the DRAM and SRAM buffers
    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);
    auto src0_l1_buffer = CreateBuffer(l1_config);
    auto src1_l1_buffer = CreateBuffer(dram_config);
    auto dst_l1_buffer = CreateBuffer(l1_config);

    // Create source data and write to DRAM
    std::vector<uint32_t> src0_vec = {14};
    std::vector<uint32_t> src1_vec = {7};

    // Enqueue write operations to copy data from host vectors to DRAM buffers. The last argument specifies whether the
    // write operation should block until the data is written to the device. In this case, we set it to false for
    // asynchronous writes, allowing the program to continue executing while the data is being written. This is
    // recommended for most writes to device in applications to improve performance.
    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, /*blocking=*/false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, /*blocking=*/false);

    // Create the kernel (code that runs on the Tensix core) that will perform the addition of the 2 integers.
    // The Data Movement cores are the only cores that can read/write data from/to DRAM. Thus we use them for
    // demonstration purposes here. In practice, you would perform addition using the compute kernel which have access
    // to much more powerful vector and matrix engines. The Data Movement cores are used for data movement tasks such as
    // reading/writing data from/to DRAM. But they are still fully capable of performing simple arithmetic operations
    // like addition. Just slower.
    KernelHandle kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Set the arguments for the kernel. The arguments are set in the order they are defined in the kernel source code.
    SetRuntimeArgs(
        program,
        kernel_id,
        core,
        {
            src0_dram_buffer->address(),
            src1_dram_buffer->address(),
            dst_dram_buffer->address(),
            src0_l1_buffer->address(),
            src1_l1_buffer->address(),
            dst_l1_buffer->address(),
        });

    // Enqueue the kernel for execution on the device. Setting blocking to false allows the program to continue
    // executing while the kernel is being executed on the device (which is for demonstration purposes here as
    // immidiately after we read the result).
    EnqueueProgram(cq, program, /*blocking=*/false);

    // Read in result into a host vector. This time we set blocking to true as we can only compare the result after the
    // data has been read from the device.
    // NOTE: Everything in the command queue is executed in order, one by one after the previous command has completed.
    // In no conditions a read on the command queue will be executed before kernel execution in front of it has
    // completed.
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, /*blocking=*/true);
    if (result_vec.size() != 1) {
        std::cout << "Error: Expected result vector size of 1, got " << result_vec.size() << std::endl;
        CloseDevice(device);
        return -1;
    }
    if (result_vec[0] != 21) {
        std::cout << "Error: Expected result of 21, got " << result_vec[0] << std::endl;
        CloseDevice(device);
        return -1;
    }

    std::cout << "Success: Result is " << result_vec[0] << std::endl;
    CloseDevice(device);
}
