#include <algorithm>
#include <functional>
#include <random>
#include <cmath>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"
#include "sfpu_helper/sfpu_helper.hpp"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

namespace unary_datacopy {
//#include "hlks/eltwise_copy.cpp"
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::uint32_t per_core_block_cnt;
    std::uint32_t per_core_block_dim;

};
}

bool run_sfpu_test(string sfpu_name) {
    bool multibank = true;
    bool pass = true;
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src_dram_buffer = ll_buda::CreateDramBuffer(device, dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 8;
        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 300 * 1024;
        uint32_t num_output_tiles = 1;
        auto cb_output = ll_buda::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto unary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            multibank ?
                "kernels/dataflow/reader_unary_8bank.cpp" :
                "kernels/dataflow/reader_unary_push_4.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            multibank ?
                "kernels/dataflow/writer_unary_8bank.cpp" :
                "kernels/dataflow/writer_unary.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new unary_datacopy::hlk_args_t{
            .per_core_block_cnt = num_tiles,
            .per_core_block_dim = 1
        };
        ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(unary_datacopy::hlk_args_t));
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        string hlk_kernel_name = "kernels/compute/eltwise_sfpu.cpp";
        auto eltwise_unary_kernel = ll_buda::CreateComputeKernel(
            program,
            hlk_kernel_name,
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        const string hlk_op_name = sfpu_op_to_hlk_op_name.at(sfpu_name);
        eltwise_unary_kernel->add_define("SFPU_OP_AND_PACK", hlk_op_name);
        bool is_relu = (sfpu_name == "relu");
        eltwise_unary_kernel->add_define("INIT_RELU", is_relu ? "hlk_relu_config(nullptr, 1);" : "");
        eltwise_unary_kernel->add_define("DEINIT_RELU", is_relu ? "hlk_relu_config(nullptr, 0);" : "");

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src_vec = sfpu_op_to_init_func.at(sfpu_name)(
            dram_buffer_size, std::chrono::system_clock::now().time_since_epoch().count());

        if (multibank)
            pass &= ll_buda::WriteToDeviceDRAMChannelsInterleavedTiles(device, src_vec, src_dram_buffer->address());
        else
            pass &= ll_buda::WriteToDeviceDRAM(src_dram_buffer, src_vec);

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {
                dram_buffer_src_addr,
                (std::uint32_t)dram_src_noc_xy.x,
                (std::uint32_t)dram_src_noc_xy.y,
                num_tiles
            }
        );

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {
                dram_buffer_dst_addr,
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                num_tiles
            }
        );

        // tt::ll_buda::tt_gdb(device, 0, program->cores(), program->cores_to_ops());
        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        if (multibank)
            ll_buda::ReadFromDeviceDRAMChannelsInterleavedTiles(
                device, dst_dram_buffer->address(), result_vec, dst_dram_buffer->size());
        else
            ll_buda::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> golden = sfpu(src_vec, sfpu_op_to_function.at(sfpu_name));

        pass &= packed_uint32_t_vector_comparison(result_vec, golden, sfpu_op_to_comparison_function.at(sfpu_name));

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles);
        }

        pass &= ll_buda::CloseDevice(device);;
        delete device;

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

    bool pass = true;
    for (const auto& [op_name, _]: sfpu_op_to_hlk_op_name) {
        log_info(LogTest, "Running {}", op_name);
        if (op_name != "sqrt") continue;
        bool pass_ = run_sfpu_test(op_name);

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
