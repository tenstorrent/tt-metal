#include <algorithm>
#include <functional>
#include <random>

#include "ll_buda/host_api.hpp"
#include "common/bfloat16.hpp"

#include "llrt/llrt.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

namespace unary_datacopy {
//#include "hlks/eltwise_copy.cpp"
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    int32_t per_core_block_cnt; // Number of blocks of size 1xN tiles (1 rows and N cols)
    int32_t per_core_block_tile_cnt; // Block tile count = (1xN)
};
}

inline vector<uint32_t> gold_standard_tilize(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {
    vector<uint32_t> dst_vec;

    int num_rows = shape.at(0);
    int num_cols = shape.at(1) / 2;
    for (int x = 0; x < num_rows; x += 32) {
        for (int y = 0; y < num_cols; y += 16) {
            int start = x * num_cols + y;

            // Top faces
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            // Bottom faces
            start += 16 * num_cols;
            for (int j = 0; j < 2; j++) {
                int start_ = start + 8 * j;
                for (int k = 0; k < 16; k++) {
                    for (int i = 0; i < 8; i++) {
                        int idx = start_ + num_cols * k + i;
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }
        }
    }

    return dst_vec;
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        ll_buda::Device *device =
            ll_buda::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= ll_buda::InitializeDevice(device);

        // ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        ll_buda::Program *program = new ll_buda::Program();

        tt_xy_pair core = {0, 0};

        uint32_t single_tile_size = 2 * 1024;

        uint32_t num_tiles_r = 1;
        uint32_t num_tiles_c = 4;
        uint32_t num_tiles = num_tiles_r * num_tiles_c;

        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src_addr = 0;
        int dram_src_channel_id = 0;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src_dram_buffer = ll_buda::CreateDramBuffer(dram_src_channel_id, dram_buffer_size, dram_buffer_src_addr);
        auto dst_dram_buffer = ll_buda::CreateDramBuffer(dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        auto dram_src_noc_xy = src_dram_buffer->noc_coordinates(device);
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates(device);

        // input CB is larger than the output CB, to test the backpressure from the output CB all the way into the input CB
        // CB_out size = 1 forces the serialization of packer and writer kernel, generating backpressure to math kernel, input CB and reader
        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 8;
        auto cb_src0 = ll_buda::CreateCircularBuffer(
            program,
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
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto unary_reader_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_unary_push_4.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_1,
            ll_buda::NOC::RISCV_1_default);

        auto unary_writer_kernel = ll_buda::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary.cpp",
            core,
            ll_buda::DataMovementProcessor::RISCV_0,
            ll_buda::NOC::RISCV_0_default);

        void *hlk_args = new unary_datacopy::hlk_args_t{
            .per_core_block_cnt = 1,
            .per_core_block_tile_cnt = (int) num_tiles_c
        };
        ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(unary_datacopy::hlk_args_t));

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/tilize.cpp",
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(
            dram_buffer_size, false);

        vector<uint32_t> golden = gold_standard_tilize(src_vec, {num_tiles_r * 32, num_tiles_c * 32});
        pass &= ll_buda::CompileProgram(device, program, skip_hlkc);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= ll_buda::WriteToDeviceDRAMChannel(device, dram_src_channel_id, src_vec, src_dram_buffer->address());

        pass &= ll_buda::ConfigureDeviceWithProgram(device, program);

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_reader_kernel,
            core,
            {dram_buffer_src_addr,
            (std::uint32_t)dram_src_noc_xy.x,
            (std::uint32_t)dram_src_noc_xy.y,
            num_tiles});

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles});


        pass &= ll_buda::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        ll_buda::ReadFromDeviceDRAMChannel(
            device, dram_dst_channel_id, dst_dram_buffer->address(), result_vec, dst_dram_buffer->size());
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        TT_ASSERT(golden.size() == result_vec.size());
        pass &= (golden == result_vec);

        if (not pass) {
            std::cout << "GOLDEN" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(golden, num_tiles);

            std::cout << "RESULT" << std::endl;
            print_vec_of_uint32_as_packed_bfloat16(result_vec, num_tiles);
        }

        pass &= ll_buda::CloseDevice(device);

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
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
