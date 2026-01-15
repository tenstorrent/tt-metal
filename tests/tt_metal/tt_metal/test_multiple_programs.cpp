// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
};

std::map<std::string, std::string> get_defines(BinaryOpType::Enum op_type) {
    std::map<std::string, std::string> defines;
    std::string op_name, op_binary_type;
    switch (op_type) {
        case BinaryOpType::ADD:
            op_name = "add_tiles";
            op_binary_type = "EltwiseBinaryType::ELWADD";
            break;
        case BinaryOpType::SUB:
            op_name = "sub_tiles";
            op_binary_type = "EltwiseBinaryType::ELWSUB";
            break;
        case BinaryOpType::MUL:
            op_name = "mul_tiles";
            op_binary_type = "EltwiseBinaryType::ELWMUL";
            break;
        default: TT_THROW("Undefined op type");
    }
    defines["ELTWISE_OP"] = op_name;
    defines["ELTWISE_OP_TYPE"] = op_binary_type;
    return defines;
}

std::tuple<Program, KernelHandle, KernelHandle> setup_program_one(
    IDevice* /*device*/, const CoreCoord& core, uint32_t single_tile_size) {
    Program program = CreateProgram();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto binary_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {};
    std::map<std::string, std::string> binary_defines = get_defines(BinaryOpType::ADD);
    binary_defines["ELTWISE_OP"] = "add_tiles";
    auto eltwise_binary_kernel = CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

    SetRuntimeArgs(program, eltwise_binary_kernel, core, {1, 1});

    return {std::move(program), binary_reader_kernel, unary_writer_kernel};
}

std::tuple<Program, KernelHandle, KernelHandle> setup_program_two(
    IDevice* /*device*/, const CoreCoord& core, uint32_t single_tile_size) {
    Program program = CreateProgram();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto mm_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_small_block.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {1, 1, 1, 1, 1, 1, 1};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    return {std::move(program), mm_reader_kernel, unary_writer_kernel};
}

void write_program_runtime_args_to_device(
    IDevice* /*device*/,
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    const CoreCoord& core,
    uint32_t num_tiles,
    Buffer& src0_dram_buffer,
    Buffer& src1_dram_buffer,
    Buffer& dst_dram_buffer) {
    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        {src0_dram_buffer.address(), (uint32_t)0, src1_dram_buffer.address(), (uint32_t)0, num_tiles});

    SetRuntimeArgs(program, writer_kernel_id, core, {dst_dram_buffer.address(), (uint32_t)0, num_tiles});
}

}  // namespace

//////////////////////////////////////////////////////////////////////////////////////////
// 1. First program runs eltwise binary on logical core {0, 0}
// 2. Host read the results from eltwise binary
// 3. Second program runs matmul, using results from step 2 as input activation
//////////////////////////////////////////////////////////////////////////////////////////
TEST_F(MeshDeviceSingleCardFixture, MultiplePrograms) {
    IDevice* dev = devices_[0]->get_devices()[0];
    CoreCoord core = {0, 0};
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;

    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    auto [program1, reader1_kernel_id, writer1_kernel_id] = setup_program_one(dev, core, single_tile_size);
    auto [program2, reader2_kernel_id, writer2_kernel_id] = setup_program_two(dev, core, single_tile_size);

    // Execute Program One
    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> src0_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::RANDOM, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto src0_activations_tile_layout =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(src0_tensor.get_values()));
    auto src0_activations = pack_bfloat16_vec_into_uint32_vec(src0_activations_tile_layout);
    detail::WriteToBuffer(src0_dram_buffer, src0_activations);

    tt::deprecated::Tensor<bfloat16> src1_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::ZEROS, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto src1_activations_tile_layout =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(src1_tensor.get_values()));
    auto src1_activations = pack_bfloat16_vec_into_uint32_vec(src1_activations_tile_layout);
    detail::WriteToBuffer(src1_dram_buffer, src1_activations);

    write_program_runtime_args_to_device(
        dev,
        program1,
        reader1_kernel_id,
        writer1_kernel_id,
        core,
        num_tiles,
        *src0_dram_buffer,
        *src1_dram_buffer,
        *dst_dram_buffer);

    detail::LaunchProgram(dev, program1);

    std::vector<uint32_t> intermediate_result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, intermediate_result_vec);

    // Validate Intermediate Result
    EXPECT_EQ(src0_activations, intermediate_result_vec) << "Eltwise binary did not produce expected result";

    // Execute Program Two - matmul with identity
    auto identity = create_identity_matrix(32, 32, 32);
    auto weights_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity));
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    detail::WriteToBuffer(src1_dram_buffer, weights);

    write_program_runtime_args_to_device(
        dev,
        program2,
        reader2_kernel_id,
        writer2_kernel_id,
        core,
        num_tiles,
        *src0_dram_buffer,
        *src1_dram_buffer,
        *dst_dram_buffer);

    detail::LaunchProgram(dev, program2);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validate - matmul with identity should give same result
    EXPECT_EQ(intermediate_result_vec, result_vec);
}
