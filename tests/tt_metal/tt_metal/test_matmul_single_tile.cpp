#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    try {
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
        const tt::ARCH arch = tt::get_arch_from_string(arch_name);
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
        uint32_t num_tiles = 1;
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_src1_addr = 0;
        int dram_src1_channel_id = 1;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src0_addr, dram_src0_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_src1_addr, dram_src1_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size, dram_buffer_dst_addr, dram_dst_channel_id, dram_buffer_size, tt_metal::BufferType::DRAM);

        auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t num_input_tiles = 2;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
        auto cb_src1 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src1_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = 2;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        auto mm_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        vector<uint32_t> compute_kernel_args = {
            1, // block_tile_dim
            1, // dst_tile_rows
            1, // dst_tile_cols
            1, // block_cnt
            1, // in0_block_tile_cnt
            1, // in1_block_tile_cnt
            1 // out_block_tile_cnt
        };
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto mm_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/matmul.cpp",
            core,
            compute_kernel_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, 32, 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto activations_tile_layout = convert_to_tile_layout(tensor.get_values());
        auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        tt_metal::WriteToBuffer(src0_dram_buffer, activations);

        auto identity = create_identity_matrix(32, 32, 32); //bflaot16 32x32 identity
        auto weights_tile_layout = convert_to_tile_layout(identity);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        tt_metal::WriteToBuffer(src1_dram_buffer, weights);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            mm_reader_kernel,
            core,
            {dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            dram_buffer_src1_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            1,
            1,
            1,
            1 * single_tile_size,
            1 * single_tile_size});

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tiles});

        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        auto result_flat_layout = convert_to_flat_layout(result_bfp16);
        pass &= (tensor.get_values() == result_flat_layout); // src1 is all 0's
        pass &= tt_metal::CloseDevice(device);;

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
