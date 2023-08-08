#include <algorithm>
#include <functional>
#include <random>
#include <cmath>

#include "third_party/magic_enum/magic_enum.hpp"

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "sfpu_helper/sfpu_helper.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
// #include "tt_gdb/tt_gdb.hpp"


// SFPU maps -> relevant kernels, golden functions, comparison functions
std::map<std::string,std::map<std::string, std::string>> sfpu_op_to_hlk_op_name={};

void update_sfpu_op_to_hlk_op()
{
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
    auto unary_op_type = magic_enum::enum_cast<tt::tt_metal::UnaryOpType>(unary_op_name).value();
    if ( tt::tt_metal::is_parametrized_type(unary_op_type) ) {
      sfpu_op_to_hlk_op_name[sfpu_op_name]  = eltwise_unary_op_utils::get_block_defines({tt::tt_metal::UnaryWithParam{unary_op_type, 0.5}});
    } else {
      sfpu_op_to_hlk_op_name[sfpu_op_name]  = eltwise_unary_op_utils::get_block_defines({tt::tt_metal::UnaryWithParam{unary_op_type, std::nullopt}});
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

bool run_sfpu_test(const tt::ARCH& arch, string sfpu_name) {

    bool multibank = true;
    bool pass = true;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(arch, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        uint32_t page_size = single_tile_size;
        if (not multibank) {
            page_size = dram_buffer_size;
        }

        auto src_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src_addr, page_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, page_size, tt_metal::BufferType::DRAM);

        auto dram_src_noc_xy = src_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 8;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src0_cb_addr
        );

        // no need for c_in2 buffer since scaler=0 in the reader kernel

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        uint32_t num_output_tiles = 1;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        auto unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ?
                "tt_metal/kernels/dataflow/reader_unary_8bank.cpp" :
                "tt_metal/kernels/dataflow/reader_unary_push_4.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            multibank ?
                "tt_metal/kernels/dataflow/writer_unary_8bank.cpp" :
                "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_tiles),
            1
        };
        string hlk_kernel_name = "tt_metal/kernels/compute/eltwise_sfpu.cpp";
        // defines macro expands per SFPU ops
        std::map<string, string> hlk_op_name = sfpu_op_to_hlk_op_name.at(sfpu_name);
        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            program,
            hlk_kernel_name,
            core,
            tt_metal::ComputeConfig{.math_approx_mode = true, .compile_args = compute_kernel_args, .defines = hlk_op_name}
        );
        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = sfpu_op_to_init_func.at(sfpu_name)(
            dram_buffer_size, std::chrono::system_clock::now().time_since_epoch().count());

        tt_metal::WriteToBuffer(src_dram_buffer, src_vec);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

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

        tt_metal::WriteRuntimeArgsToDevice(device, program);

        // tt::tt_metal::tt_gdb(device, 0, program->cores(), program->cores_to_ops());
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> golden = sfpu(src_vec, sfpu_op_to_function.at(sfpu_name));

        pass &= packed_uint32_t_vector_comparison(result_vec, golden, sfpu_op_to_comparison_function.at(sfpu_name));

        if (not pass) {
            // Printing of large tiles causes a system lockup. Do not print these unless debugging please.
            //std::cout << "GOLDEN" << std::endl;
            //print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles);

            //std::cout << "RESULT" << std::endl;
            //print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles);
        }

        pass &= tt_metal::CloseDevice(device);;
        // TODO (abhullar): Uplift when raw ptr usages are removed. Commenting out delete for now because device needs to outlive buffers
        //delete device;

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_fatal(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

int main(int argc, char **argv) {

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    string arch_name = "";
    try {
        std::tie(arch_name, input_args) =
            test_args::get_command_option_and_remaining_args(input_args, "--arch", "grayskull");
    } catch (const std::exception& e) {
        log_fatal(tt::LogTest, "Command line arguments found exception", e.what());
    }
    update_sfpu_op_to_hlk_op();
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    for (const auto& [op_name, _]: sfpu_op_to_hlk_op_name) {
        log_info(LogTest, "Running {}", op_name);

        bool pass_ = run_sfpu_test(arch, op_name);

        if (pass_) {
            log_info(LogTest, "{} test passed", op_name);
        } else {
            log_info(LogTest, "{} test failed", op_name);
        }

        pass &= pass_;
    }

    if (pass) {
        log_info(LogTest, "Sfpu tests passed");
    } else {
        log_fatal(LogTest, "Sfpu tests failed");
    }

    return 0;
}
