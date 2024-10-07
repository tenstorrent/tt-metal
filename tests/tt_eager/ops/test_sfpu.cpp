// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <sstream>

#include <magic_enum.hpp>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/buffers/buffer.hpp"
#include "common/bfloat16.hpp"
#include "tests_common/sfpu_helper/sfpu_helper.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "tt_metal/impl/program/program_pool.hpp"
// #include "tt_gdb/tt_gdb.hpp"


// SFPU maps -> relevant kernels, golden functions, comparison functions
std::map<std::string,std::map<std::string, std::string>> sfpu_op_to_hlk_op_name={};

void update_sfpu_op_to_hlk_op()
{
    using ttnn::operations::unary::UnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;

    for(const std::string& sfpu_op_name : sfpu_op) {
        std::string unary_op_name{sfpu_op_name};
        for(auto& c: unary_op_name) {
            c = toupper(c);
        }
        if (unary_op_name == "EXPONENTIAL") {
            unary_op_name = "EXP";
        }
        else if (unary_op_name == "RECIPROCAL") {
            unary_op_name = "RECIP";
        }
        auto unary_op_type = magic_enum::enum_cast<UnaryOpType>(unary_op_name).value();
        if (ttnn::operations::unary::utils::is_parametrized_type(unary_op_type)) {
            if (unary_op_type == UnaryOpType::EXP) {
                sfpu_op_to_hlk_op_name[sfpu_op_name]  = ttnn::operations::unary::utils::get_block_defines({UnaryWithParam{unary_op_type, 1.0}});
            } else {
                sfpu_op_to_hlk_op_name[sfpu_op_name]  = ttnn::operations::unary::utils::get_block_defines({UnaryWithParam{unary_op_type, 0.5}});
            }
        } else {
            sfpu_op_to_hlk_op_name[sfpu_op_name]  = ttnn::operations::unary::utils::get_block_defines({UnaryWithParam{unary_op_type}});
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool run_sfpu_test(string sfpu_name,int tile_factor=1,bool use_DRAM=true) {

    bool multibank = true;
    bool pass = true;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        auto program = tt_metal::CreateScopedProgram();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t page_size = single_tile_size;
        if (not multibank) {
            page_size = dram_buffer_size;
        }

        tt_metal::BufferType buffType = (use_DRAM) ? tt_metal::BufferType::DRAM : tt_metal::BufferType::L1;
        tt_metal::InterleavedBufferConfig buff_config{
                                        .device=device,
                                        .size = dram_buffer_size,
                                        .page_size = page_size,
                                        .buffer_type = buffType
                                        };

        auto src_dram_buffer = CreateBuffer(buff_config);
        uint32_t dram_buffer_src_addr = src_dram_buffer->address();
        auto dst_dram_buffer = CreateBuffer(buff_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 8;
        tt_metal::CircularBufferConfig src_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, src_cb_config);

        // no need for c_in2 buffer since scaler=0 in the reader kernel

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 1;
        tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, output_cb_config);

        auto unary_reader_kernel = tt_metal::CreateKernel(
                                                          program,
                                                          multibank ?
                                                          "tests/tt_eager/kernels/dataflow/reader_unary_8bank.cpp" :
                                                          "tests/tt_eager/kernels/dataflow/reader_unary_push_4.cpp",
                                                          core,
                                                          tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
                                                          program,
                                                          multibank ?
                                                          "tests/tt_eager/kernels/dataflow/writer_unary_8bank.cpp" :
                                                          "tt_metal/kernels/dataflow/writer_unary.cpp",
                                                          core,
                                                          tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
                                                (uint)num_tiles,
                                                1,
                                                (uint)tile_factor,
        };
        string hlk_kernel_name = "tests/tt_eager/ops/kernel/eltwise_sfpu.cpp";

        // defines macro expands per SFPU ops
        std::map<string, string> hlk_op_name = sfpu_op_to_hlk_op_name.at(sfpu_name);
        auto eltwise_unary_kernel = tt_metal::CreateKernel(
                                                           program,
                                                           hlk_kernel_name,
                                                           core,
                                                           tt_metal::ComputeConfig{.math_approx_mode = true, .compile_args = compute_kernel_args, .defines = hlk_op_name}
                                                           );
        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////



        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = sfpu_op_to_init_func.at(sfpu_name)(
                                                                           dram_buffer_size, std::chrono::system_clock::now().time_since_epoch().count());

        tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);



        tt_metal::SetRuntimeArgs(
                                 program,
                                 unary_reader_kernel,
                                 core,
                                 {
                                  dram_buffer_src_addr,
                                  (std::uint32_t)dram_src_noc_xy.x,
                                  (std::uint32_t)dram_src_noc_xy.y,
                                  num_tiles,
                                  0,0,0,0,0 // TODO(AP): [8] is scaler
                                 }
                                 );

        tt_metal::SetRuntimeArgs(
                                 program,
                                 unary_writer_kernel,
                                 core,
                                 {
                                  dram_buffer_dst_addr,
                                  (std::uint32_t)dram_dst_noc_xy.x,
                                  (std::uint32_t)dram_dst_noc_xy.y,
                                  num_tiles
                                 }
                                 );


        auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
        tt_metal::detail::LaunchProgram(device, *program_ptr);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> golden = sfpu(src_vec, sfpu_op_to_function.at(sfpu_name));

        pass &= packed_uint32_t_vector_comparison(result_vec, golden, sfpu_op_to_comparison_function.at(sfpu_name));


        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        TT_THROW("System error message: {}", std::strerror(errno));
    }

    return pass;
}

bool run_unit_test(std::string op_name,int tile_factor,bool use_DRAM) {
    log_info(LogTest, "Running {}", op_name);

    bool pass_ = run_sfpu_test(op_name,tile_factor,use_DRAM);

    if (pass_) {
        log_info(LogTest, "{} test passed", op_name);
    } else {
        log_info(LogTest, "{} test failed", op_name);
    }
    return pass_;
}

int main(int argc, char **argv) {
    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    bool pass = true;
    int arg_tile_factor = 1;
    int arg_use_DRAM = true;
    int arg_help = false;
    update_sfpu_op_to_hlk_op();


    if ( argc == 1 ) {
        for (const auto& [op_name, _]: sfpu_op_to_hlk_op_name) {
            pass &= run_unit_test(op_name,arg_tile_factor, arg_use_DRAM);
            if (pass)
                log_info(LogTest, "PASS-SFPU test {}",op_name);
            else
                log_info(LogTest, "FAIL-SFPU test {}",op_name);
        }
    } else {
        std::vector<std::string> operators;
        for(uint32_t idx = 1; idx < argc; idx++) {
            if ( strstr(argv[idx],"-tile-factor") ) {
                idx++;
                arg_tile_factor = atoi(argv[idx]);
            } else if( strstr(argv[idx],"-use-L1") ) {
                arg_use_DRAM = false;
            } else if( strstr(argv[idx],"-use-DRAM") ) {
                arg_use_DRAM = true;
            } else if( strstr(argv[idx],"-help") ) {
                arg_help = true;
                break;
            } else {
                operators.push_back(std::string(argv[idx]));
            }
        }
        if ( arg_help ) {
            std::stringstream ss;
            ss << "Usage: test_sfpu {operators}+ [flags]+\n";
            ss <<"--use-L1 or --use-DRAM chooses between L1 or DRAM. Default is DRAM\n";
            ss << "--tile-factor: integer between 1 to 1024 to specify number of repetitions of 32x64x32x32 tensor\n";
            ss << "operators are standard SFPU operators:\n\t";
            for (const auto& [op_name, _]: sfpu_op_to_hlk_op_name) {
                ss << op_name << ", ";
            }
            log_info(LogTest,"Help: {}",ss.str().c_str());
            exit(0);
        }
        for(uint32_t idx = 0; idx < operators.size(); idx++) {
            pass &= run_unit_test(operators[idx],arg_tile_factor,arg_use_DRAM);
        }
    }

    if (pass) {
        log_info(LogTest, "Sfpu tests passed");
    } else {
        TT_THROW("Sfpu tests failed");
    }

    return 0;
}
