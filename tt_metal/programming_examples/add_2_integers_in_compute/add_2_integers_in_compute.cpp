// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>

using namespace tt;
using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main() {
    /* Silicon accelerator setup */
    IDevice* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    // Since all interleaved buffers have size == page_size, they are entirely contained in the first DRAM bank
    uint32_t src0_bank_id = 0;
    uint32_t src1_bank_id = 0;
    uint32_t dst_bank_id = 0;

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t output_cb_index = CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/reader_binary_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/dataflow/writer_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "add_2_integers_in_compute/kernels/compute/add_2_tiles.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src0_vec;
    std::vector<uint32_t> src1_vec;
    src0_vec = create_constant_vector_of_bfloat16(single_tile_size, 14.0f);
    src1_vec = create_constant_vector_of_bfloat16(single_tile_size, 8.0f);

    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {src0_dram_buffer->address(), src1_dram_buffer->address(), src0_bank_id, src1_bank_id});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_bank_id});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    printf("Result = %d\n", result_vec[0]);  // 22 = 1102070192
    printf(
        "Expected = %d\n",
        pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat16(22.0f), bfloat16(22.0f))));
    CloseDevice(device);
}
