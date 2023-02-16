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
    int32_t per_core_tile_cnt; // Total number of tiles produced at the output per core
    int32_t per_core_block_cnt; // Number of blocks of size 1xN tiles (1 rows and N cols)
    int32_t per_core_block_tile_cnt; // Block tile count = (1xN)
};
}

uint32_t prod(vector<uint32_t> &shape) {
    uint32_t shape_prod = 1;

    for (uint32_t shape_i: shape) {
        shape_prod *= shape_i;
    }

    return shape_prod;
}

inline vector<uint32_t> gold_standard_untilize(std::vector<uint32_t> src_vec, vector<uint32_t> shape) {
    vector<uint32_t> dst_vec;

    int num_rows = shape.at(0);
    int num_cols = shape.at(1) / 2;

    int face_size = 16 * 8;
    int tile_size = face_size * 4;

    // Iterate over tile rows
    for (int t = 0; t < num_rows / 32; t++) {

        int tile_start_index = t * (num_cols / 16);

        int physical_start_for_tile_row = tile_start_index * 32;

        // Iterate over tile columns 32 times (naive, but simple for validation)
        for (int x = 0; x < 2; x++) {
            for (int i = 0; i < 16; i++) { // num rows in a face
                for (int j = 0; j < num_cols / 16; j++) { // num columns top two faces
                    // Left face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + i * 8 + k + j * tile_size;
                        dst_vec.push_back(src_vec.at(idx));
                    }

                    // Right face row copy
                    for (int k = 0; k < 8; k++) {
                        int idx = physical_start_for_tile_row + i * 8 + k + face_size + j * tile_size; 
                        dst_vec.push_back(src_vec.at(idx));
                    }
                }
            }

            physical_start_for_tile_row += 2 * face_size; // Move to bottom faces
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
            "kernels/dataflow/reader_unary.cpp",
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
            .per_core_tile_cnt = (int) num_tiles,
            .per_core_block_cnt = 1,
            .per_core_block_tile_cnt = (int) num_tiles_c
        };
        ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(unary_datacopy::hlk_args_t));
        
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = ll_buda::CreateComputeKernel(
            program,
            "kernels/compute/untilize.cpp",
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

        vector<uint32_t> golden = gold_standard_untilize(src_vec, {num_tiles_r * 32, num_tiles_c * 32});
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
