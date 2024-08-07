// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/detail/tt_metal.hpp"
using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char **argv) {

    // Initialize Program and Device
    constexpr CoreCoord core = {5, 0};
    int device_id = 0;
    Device *device = CreateDevice(device_id);
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr uint32_t single_tile_size = 1024;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> collect_dram_buffer = CreateBuffer(dram_config);

    constexpr uint32_t src0_cb_index = CB::c_in0;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}}).set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    std::vector<uint32_t> initial_int_value(1, 0);
    EnqueueWriteBuffer(cq, collect_dram_buffer, initial_int_value, false);
    KernelHandle int_num_gen_and_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/one_to_one_datastream/kernels/dataflow/int_num_gen_and_reader_kernel.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    SetRuntimeArgs(program, int_num_gen_and_reader_kernel_id, core, {collect_dram_buffer->address()});
    EnqueueProgram(cq, program, false);
    Finish(cq);
    CloseDevice(device);
    return 0;
}
