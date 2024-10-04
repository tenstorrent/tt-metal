// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "core_coord.h"
#include "tt_metal/common/bfloat16.hpp"
// #include "tt_metal/common/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
// #include "ttnn/cpp/ttnn/tensor/tensor.hpp"
// #include "ttnn/cpp/ttnn/tensor/types.hpp"

using namespace tt;
using namespace tt::tt_metal;

// void create(const Tensor &input, const ttnn::DeviceComputeKernelConfig compute_kernel_config) {
//     Device *device = input.device();

//     auto grid = CoreCoord(0, 0);

//     uint32_t units_to_divide = input.volume() / constants::TILE_HEIGHT / constants::TILE_WIDTH;
//     auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
//         split_work_to_cores(grid, units_to_divide);

//     auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc] =
//         get_compute_kernel_config_args(device->arch(), compute_kernel_config);

//     Program program = Program();

//     tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());

//     // tt::operations::primary::CreateCircularBuffer(program, all_cores, data_format, {{CB::c_in0, 2, data}});
// }

int main(int argc, char **argv) {
    /* Silicon accelerator setup */
    Device *device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue &cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 4 * 1024;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    // std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    // auto src0_dram_noc_coord = src0_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
    // uint32_t src0_dram_noc_x = src0_dram_noc_coord.x;
    // uint32_t src0_dram_noc_y = src0_dram_noc_coord.y;
    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t src0_cb_index = CB::c_in0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Int32}})
            .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t output_cb_index = CB::c_out0;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float32}})
            .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/uniform/kernels/reader.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/uniform/kernels/writer.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Set the parameters that the compute kernel will use */
    vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "tt_metal/programming_examples/uniform/kernels/uniform.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = true,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src0_vec(1024);
    std::vector<uint32_t> src1_vec(1024);

    // EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
    // EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(program, binary_reader_kernel_id, core, {});
    // {src0_dram_buffer->address(),
    //  src1_dram_buffer->address(),
    //  src0_dram_noc_x,
    //  src0_dram_noc_y,
    //  src1_dram_noc_x,
    //  src1_dram_noc_y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_dram_noc_x, dst_dram_noc_y});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<float> result_vec(1024);
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec.data(), true);
    std::map<float, int> mp;

    for (uint32_t i = 0; i < 1024; ++i) {
        float a = result_vec[i];
        mp[result_vec[i]] += 1;
        std::cout << result_vec[i] << " ";
        if ((i & 31) == 31)
            std::cout << std::endl;
    }

    std::cout << mp.size() << std::endl;
    // for (const auto &pair : mp) {
    //     std::cout << std::bitset<32>(pair.first) << " " << pair.second << std::endl;
    // }

    CloseDevice(device);
}
