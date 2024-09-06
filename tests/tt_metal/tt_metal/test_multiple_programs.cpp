// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"

using namespace tt;

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

std::map<string, string> get_defines(BinaryOpType::Enum op_type){
    // TODO(AP): remove duplication
    std::map<string, string> defines;
    string op_name, op_binary_type;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; op_binary_type = "EltwiseBinaryType::ELWADD"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; op_binary_type = "EltwiseBinaryType::ELWSUB"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; op_binary_type = "EltwiseBinaryType::ELWMUL"; break;
        default: TT_FATAL(false && "Undefined op type");
    }
    defines["ELTWISE_OP"] = op_name.c_str();
    defines["ELTWISE_OP_TYPE"] = op_binary_type.c_str();
    return defines;
}


std::tuple<tt_metal::Program, tt_metal::KernelHandle, tt_metal::KernelHandle> setup_program_one(tt_metal::Device *device, const CoreCoord &core, uint32_t single_tile_size) {
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto binary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
    };
    std::map<string, string> binary_defines = get_defines(BinaryOpType::ADD);
    binary_defines["ELTWISE_OP"] = "add_tiles";
    auto eltwise_binary_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines}
    );

    SetRuntimeArgs(
        program,
        eltwise_binary_kernel,
        core,
        {1, 1}
    );

    return {std::move(program), binary_reader_kernel, unary_writer_kernel};
}

std::tuple<tt_metal::Program, tt_metal::KernelHandle, tt_metal::KernelHandle> setup_program_two(tt_metal::Device *device, const CoreCoord &core, uint32_t single_tile_size) {
    tt_metal::Program program = tt_metal::CreateProgram();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto mm_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_small_block.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        1, // block_tile_dim
        1, // dst_tile_rows
        1, // dst_tile_cols
        1, // block_cnt
        1, // in0_block_tile_cnt
        1, // in1_block_tile_cnt
        1 // out_block_tile_cnt
    };

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    return {std::move(program), mm_reader_kernel, unary_writer_kernel};
}

void write_program_runtime_args_to_device(
    tt_metal::Device *device,
    tt_metal::Program &program,
    tt_metal::KernelHandle reader_kernel_id,
    tt_metal::KernelHandle writer_kernel_id,
    const CoreCoord &core,
    uint32_t num_tiles,
    tt_metal::Buffer &src0_dram_buffer,
    tt_metal::Buffer &src1_dram_buffer,
    tt_metal::Buffer &dst_dram_buffer) {

    auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();


    tt_metal::SetRuntimeArgs(
        program, reader_kernel_id, core,
        {src0_dram_buffer.address(),
        (std::uint32_t)dram_src0_noc_xy.x,
        (std::uint32_t)dram_src0_noc_xy.y,
        src1_dram_buffer.address(),
        (std::uint32_t)dram_src1_noc_xy.x,
        (std::uint32_t)dram_src1_noc_xy.y,
        num_tiles}
    );

    tt_metal::SetRuntimeArgs(
        program, writer_kernel_id, core,
        {dst_dram_buffer.address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        num_tiles}
    );


}
//////////////////////////////////////////////////////////////////////////////////////////
// 1. First program runs eltwise binary on logical core {0, 0}
// 2. Host read the results from eltwise binary
// 3. Second program runs matmul, using results from step 2 as input activation
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;

        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = dram_buffer_size,
                    .page_size = dram_buffer_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };


        auto src0_dram_buffer = CreateBuffer(dram_config);
        auto src1_dram_buffer = CreateBuffer(dram_config);
        auto dst_dram_buffer = CreateBuffer(dram_config);

        auto [program1, reader1_kernel_id, writer1_kernel_id] = setup_program_one(device, core, single_tile_size);

        auto [program2, reader2_kernel_id, writer2_kernel_id] = setup_program_two(device, core, single_tile_size);

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Applications
        ////////////////////////////////////////////////////////////////////////////



        // Both programs use the same CB addresses but they can be compiled one after
        // the other because they use the same data formats


        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Program One
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, 32, 32};
        tt::deprecated::Tensor<bfloat16> src0_tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto src0_activations_tile_layout = convert_to_tile_layout(src0_tensor.get_values());
        auto src0_activations = pack_bfloat16_vec_into_uint32_vec(src0_activations_tile_layout);
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_activations);

        tt::deprecated::Tensor<bfloat16> src1_tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::ZEROS, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto src1_activations_tile_layout = convert_to_tile_layout(src1_tensor.get_values());
        auto src1_activations = pack_bfloat16_vec_into_uint32_vec(src1_activations_tile_layout);
        tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_activations);



        write_program_runtime_args_to_device(device, program1, reader1_kernel_id, writer1_kernel_id, core, num_tiles, *src0_dram_buffer, *src1_dram_buffer, *dst_dram_buffer);

        tt_metal::detail::LaunchProgram(device, program1);

        std::vector<uint32_t> intermediate_result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, intermediate_result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validatie Intermediate Result
        ////////////////////////////////////////////////////////////////////////////
        pass &= (src0_activations == intermediate_result_vec);  // src1 is ZEROS
        if (pass) {
            log_info(LogTest, "Eltwise binary ran successfully");
        } else {
            log_error(LogTest, "Eltwise binary did not run sucessfully!");
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Program Two
        ////////////////////////////////////////////////////////////////////////////
        // Write matmul weights to DRAM
        auto identity = create_identity_matrix(32, 32, 32); //bflaot16 32x32 identity
        auto weights_tile_layout = convert_to_tile_layout(identity);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        tt_metal::detail::WriteToBuffer(src1_dram_buffer, weights);



        write_program_runtime_args_to_device(device, program2, reader2_kernel_id, writer2_kernel_id, core, num_tiles, *src0_dram_buffer, *src1_dram_buffer, *dst_dram_buffer);

        tt_metal::detail::LaunchProgram(device, program2);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        pass &= (intermediate_result_vec == result_vec); // src1 is identity matrix

        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
