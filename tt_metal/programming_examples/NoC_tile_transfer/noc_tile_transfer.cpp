// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>

#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

int main() {
    // Device setup
    IDevice* device = CreateDevice(0);

    // Device command queue and program setup
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    // Core range setup
    constexpr CoreCoord core0 = {0, 0};
    constexpr CoreCoord core1 = {0, 1};
    const auto core0_physical_coord = device->worker_core_from_logical_core(core0);
    const auto core1_physical_coord = device->worker_core_from_logical_core(core1);

    CoreRange sem_core_range = CoreRange(core0, core1);

    // Check if the environment variable for kernels print is set
    char* env_var = std::getenv("TT_METAL_DPRINT_CORES");
    if (env_var == nullptr) {
        std::cerr
            << "WARNING: Please set the environment variable TT_METAL_DPRINT_CORES to (0,0),(0,1) to see the output of "
               "the Data Movement kernels. Command: export TT_METAL_DPRINT_CORES=(0,0),(0,1)"
            << std::endl;
    }

    // Input data preparation
    constexpr uint32_t single_tile_size = sizeof(uint16_t) * tt::constants::TILE_HW;
    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};

    std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);  // Input buffer
    std::shared_ptr<Buffer> dst_dram_buffer = CreateBuffer(dram_config);  // Output buffer

    const bool input_tensor_is_dram = src_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_tensor_is_dram = dst_dram_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // Core synchronization semaphore setup
    const uint32_t sem_id = CreateSemaphore(program, sem_core_range, 0);

    // Source data preparation and DRAM transfer
    const uint16_t input_data = 14;  // Example input data
    std::vector<uint16_t> src_vec(1, input_data);
    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec.data(), false);

    // L1 circular buffer setup
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::UInt16}})
            .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(single_tile_size, {{src1_cb_index, tt::DataFormat::UInt16}})
            .set_page_size(src1_cb_index, single_tile_size);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, sem_core_range, cb_src1_config);

    // Kernels setup
    // Core 0 kernels
    KernelHandle core0_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader0.cpp",
        core0,
        tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, static_cast<uint32_t>(input_tensor_is_dram)}});
    KernelHandle core0_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer0.cpp",
        core0,
        tt::tt_metal::WriterDataMovementConfig{{src0_cb_index, src1_cb_index}});

    // Core 1 kernels
    KernelHandle core1_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/reader1.cpp",
        core1,
        tt::tt_metal::ReaderDataMovementConfig{{src0_cb_index, src1_cb_index}});
    KernelHandle core1_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "NoC_tile_transfer/kernels/dataflow/writer1.cpp",
        core1,
        tt::tt_metal::WriterDataMovementConfig{{src1_cb_index, static_cast<uint32_t>(output_tensor_is_dram)}});

    // Runtime args setup
    SetRuntimeArgs(program, core0_reader_kernel_id, core0, {src_dram_buffer->address()});
    SetRuntimeArgs(program, core0_writer_kernel_id, core0, {core1_physical_coord.x, core1_physical_coord.y, sem_id});
    SetRuntimeArgs(program, core1_reader_kernel_id, core1, {core0_physical_coord.x, core0_physical_coord.y, sem_id});
    SetRuntimeArgs(program, core1_writer_kernel_id, core1, {dst_dram_buffer->address()});

    // Program enqueue
    EnqueueProgram(cq, program, false);
    Finish(cq);

    // Data transfer back to host machine
    std::vector<uint16_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);  // Blocking call to ensure data is read before proceeding

    std::cout << "Result = " << result_vec[0] << " : Expected = " << input_data << std::endl;

    CloseDevice(device);
}
