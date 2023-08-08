//////////////////////////////////////////////////////////////////////////////////////////
// Tests a 3x3 convolution by implementing as a matmul.
// Converts the layout of activation from nchw to nhwc on the host and copies to DRAM.
// Converts the layout of weights to a 2d matrix and tilizes it on the host before copying to DRAM.
// Computes an address map on the host to copy the untilized activations from DRAM and tilize them in L1.
// Uses "generic_binary_reader_blocked" kernel to read untilized activations from DRAM using the address map computed on host.
// The "generic_binary_reader_blocked" kernel also reads the tilized weights from DRAM.
// Uses "matmul_large_block_zm" kernel to do the compute. Uses "writer_unswizzle" kernel to write tilized output to DRAM.
//////////////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <functional>
#include <random>
#include<chrono>
#include <tuple>
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "common/constants.hpp"
// This file contains helper functions to do layout transformations (tilize, untilize) and
// to compute the address map for copying activations from DRAM to L1
#include "tt_metal/llrt/test_libs/conv_pattern.hpp"
#include "llrt/tt_debug_print_server.hpp"
using namespace tt;
using namespace tt::constants;

int main(int argc, char **argv) {
    bool pass = true;

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

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
        ConvParameters conv_params = ConvParameters(3, 3, 1, 1, 0, 0);
        std::array<uint32_t, 4> act_shape = {1, 64, 10, 10};
        std::array<uint32_t, 4> weight_shape = {576, 64, 3, 3};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(act_shape, tt::deprecated::Initialize::RANDOM, std::chrono::system_clock::now().time_since_epoch().count());
        std::array<std::array<uint32_t, 2>, 4> pad_size = {{{0, 0}, {0, 0}, {conv_params.PadH, conv_params.PadH}, {conv_params.PadW, conv_params.PadW}}};
        bfloat16 pad_value = (uint32_t) 0;
        // convolution input is padded on the host. TODO: padding should be moved to device reader kernel
        tt::deprecated::Tensor<bfloat16> tensor_padded = tt::deprecated::pad(tensor, pad_size, pad_value);
        auto tensor_p = tt::deprecated::permute(tensor_padded, {0, 2, 3, 1}); // NHWC
        // Overwrite the weight tensor with identity matrix after intializing it.
        tt::deprecated::Tensor<bfloat16> weight_tensor = tt::deprecated::initialize_tensor<bfloat16>(weight_shape, tt::deprecated::Initialize::ZEROS);
        auto weight_tensor_p = tt::deprecated::permute(weight_tensor, {0, 2, 3, 1}); // NHWC

        // generate address map to generic reader kernel
        std::tuple<uint32_t, uint32_t, uint32_t, std::vector<uint32_t>> addr_ = gen_source_addresses_for_conv_act_layout_transform(tensor_p.get_shape(), conv_params, sizeof(bfloat16));
        auto num_tiles_generated_with_source_addresses = std::get<0>(addr_);
        auto num_addresses_per_tile = std::get<1>(addr_);
        auto dram_read_size_bytes = std::get<2>(addr_);
        auto source_addresses = std::get<3>(addr_);
        // The source addresses are addresses for convolution activation in DRAM
        // It is used by the generic reader kernel. The source addresses are arranged in the order of tiles.
        // The dram read size is fixed to 16 elements which is one row of face within a tile .
        // The kernel determines the L1 address as it writes to contingous locations in L1 buffer.

        // vector to be copied to DRAM
        auto src_vec = tensor_p.get_values();
        // This will create the 2D matrix by modeling what dram to l1 read patterns are
        auto golden_act_matrix_tilized = move_act_dram_to_l1_tilized(tensor_p, dram_read_size_bytes, source_addresses);
        // This would be the actual golden that we compare the activation data
        auto golden_act_matrix = move_act_dram_to_l1(tensor_p, conv_params);
        auto golden_act_vector = flatten(golden_act_matrix);
        std::uint32_t act_rows = golden_act_matrix.size();
        std::uint32_t act_cols = golden_act_matrix.at(0).size();
        // Sanity check to verify address map.
        auto golden_act_untilized = untilize_act(golden_act_matrix_tilized, act_rows, act_cols);
        assert(golden_act_vector == golden_act_untilized);
        auto weight_matrix_ = move_weights_dram_to_l1_mm(weight_tensor_p);
        std::uint32_t weight_rows = weight_matrix_.size();
        std::uint32_t weight_cols = weight_matrix_.at(0).size();
        // For zero weight test -
        //auto weight_vector = flatten(weight_matrix_);
        // For identity test - Creating a new identity weight matrix
        auto weight_vector = create_identity_matrix(weight_rows, weight_cols, std::min(weight_rows, weight_cols));
        // tilize weights to be copied to DRAM
        // TODO: should we do this on device when reading from DRAM to L1?
        auto weights_tilized = tilize(weight_vector, weight_rows, weight_cols);
        std::array<uint32_t, 4> output_shape = {1, 1, act_rows, weight_cols};
        // For identity test -
        auto golden_output_vec = golden_act_vector;
        // For zero weight test -
        //tt::deprecated::Tensor<bfloat16> golden_output_tensor = tt::deprecated::initialize_tensor<bfloat16>(output_shape, tt::deprecated::Initialize::ZEROS);
        //auto golden_output_vec = golden_output_tensor.get_values();

        uint32_t single_tile_size = 2 * 1024;
        assert(act_rows % TILE_HEIGHT == 0);
        assert(act_cols % TILE_WIDTH == 0);
        assert(weight_rows % TILE_HEIGHT == 0);
        assert(weight_cols % TILE_WIDTH == 0);
        std::uint32_t num_tiles_rows = act_rows / TILE_HEIGHT;
        std::uint32_t num_tiles_cols = act_cols / TILE_WIDTH;
        std::uint32_t num_tiles = num_tiles_rows * num_tiles_cols;
        std::uint32_t w_num_tiles_rows = weight_rows / TILE_HEIGHT;
        std::uint32_t w_num_tiles_cols = weight_cols / TILE_WIDTH;
        std::uint32_t w_num_tiles = w_num_tiles_rows * w_num_tiles_cols;
        assert(act_cols == weight_rows);
        assert(num_tiles == num_tiles_generated_with_source_addresses);
        uint32_t output_rows = act_rows;
        uint32_t output_cols = weight_cols;
        uint32_t output_tiles_rows = num_tiles_rows;
        uint32_t output_tiles_cols = w_num_tiles_cols;


        uint32_t M = num_tiles_rows;
        uint32_t K = num_tiles_cols;
        uint32_t N = w_num_tiles_cols;
        int out_subblock_h = 2;
        int out_subblock_w = 3;
        int in0_block_w = 1;


        int num_blocks = K/in0_block_w;
        uint32_t src0_num_tiles_per_block = M * in0_block_w;
        uint32_t src1_num_tiles_per_block = N * in0_block_w;
        // src0_num_reads_per_block is the number of DRAM reads issued to produce 1 block
        uint32_t src0_num_reads_per_block = src0_num_tiles_per_block * num_addresses_per_tile;
        assert(source_addresses.size() == num_blocks * src0_num_reads_per_block);
        uint32_t src0_num_bytes_per_block = src0_num_tiles_per_block * single_tile_size;
        uint32_t src1_num_bytes_per_block = src1_num_tiles_per_block * single_tile_size;

        tt_metal::Program program = tt_metal::Program();
        CoreCoord core = {0, 0};
        uint32_t dram_buffer_src0_size = tensor_p.get_volume() * sizeof(bfloat16);
        uint32_t dram_buffer_src1_size = weights_tilized.size() * sizeof(bfloat16);
        uint32_t dram_buffer_dst_size = M * N * single_tile_size;

        uint32_t dram_buffer_src0_addr = 0;
        uint32_t dram_buffer_src1_addr = dram_buffer_src0_addr + dram_buffer_src0_size;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)

        auto src0_dram_buffer = tt_metal::Buffer(device, dram_buffer_src0_size, dram_buffer_src0_addr, dram_buffer_src0_size, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = tt_metal::Buffer(device, dram_buffer_src1_size, dram_buffer_src1_addr, dram_buffer_src1_size, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_dst_size, dram_buffer_dst_addr, dram_buffer_dst_size, tt_metal::BufferType::DRAM);

        auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        uint32_t cb0_index = 0;
        uint32_t cb0_addr = 200 * 1024;
        uint32_t num_cb0_tiles = M * in0_block_w * 2;
        uint32_t cb0_size = num_cb0_tiles * single_tile_size;
        uint32_t source_addresses_in_l1_addr = cb0_addr + cb0_size;
        auto cb0 = tt_metal::CreateCircularBuffer(
            program,
            cb0_index,
            core,
            num_cb0_tiles,
            cb0_size,
            tt::DataFormat::Float16_b,
            cb0_addr
        );

        uint32_t cb1_index = 1;
        uint32_t cb1_addr = 300 * 1024;
        uint32_t num_cb1_tiles = N * in0_block_w * 2;
        uint32_t cb1_size = num_cb1_tiles * single_tile_size;
        auto cb1 = tt_metal::CreateCircularBuffer(
            program,
            cb1_index,
            core,
            num_cb1_tiles,
            cb1_size,
            tt::DataFormat::Float16_b,
            cb1_addr
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t interm0_cb_index = 24;
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = M * N;
        uint32_t cb_output_size = num_output_tiles * single_tile_size;
        CoreRangeSet cores(std::set<CoreRange>{CoreRange{.start=core, .end=core}});
        auto cb_output = tt_metal::CreateCircularBuffers(
            program,
            {ouput_cb_index, interm0_cb_index},
            cores,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        std::vector<uint32_t> generic_binary_reader_args {
            dram_buffer_src0_addr,
            (uint32_t)dram_src0_noc_xy.x,
            (uint32_t)dram_src0_noc_xy.y,
            dram_buffer_src1_addr,
            (uint32_t)dram_src1_noc_xy.x,
            (uint32_t)dram_src1_noc_xy.y,
            (uint32_t)source_addresses.size(),
            (uint32_t)source_addresses_in_l1_addr,
            (uint32_t)num_blocks,
            src0_num_reads_per_block,
            dram_read_size_bytes,
            src1_num_bytes_per_block,
            src0_num_tiles_per_block,
            src1_num_tiles_per_block};

        auto generic_binary_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/generic_binary_reader_blocked.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        std::vector<uint32_t> writer_rt_args{
            dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            (std::uint32_t)out_subblock_h, // num tiles per sub block m
            (std::uint32_t)out_subblock_w, // num tiles per sub block n
            (std::uint32_t)M/out_subblock_h, // num sub blocks m
            (std::uint32_t)N/out_subblock_w, // num sub blocks n
            (std::uint32_t)out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row within sub-block
            (std::uint32_t)out_subblock_h * out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row of sub-blocks
            (std::uint32_t)out_subblock_w*single_tile_size}; // bytes offset to next sub-block

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unswizzle.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        int in0_num_subblocks = (M/out_subblock_h);
        int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
        int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        int in1_num_subblocks = (N/out_subblock_w);
        int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
        int in1_per_core_w = out_subblock_w * in1_num_subblocks;

        int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

        vector<uint32_t> compute_kernel_args = {
            uint(in0_block_w),
            uint(in0_num_subblocks),
            uint(in0_block_num_tiles),
            uint(in0_subblock_num_tiles),

            uint(in1_num_subblocks),
            uint(in1_block_num_tiles),
            uint(in1_per_core_w),

            uint(num_blocks),

            uint(out_subblock_h),
            uint(out_subblock_w),
            uint(out_subblock_num_tiles)
        };

        auto mm_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/matmul_large_block_zm.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        std::cout << "GOING TO COMPILE PROGRAM." << std::endl;
        pass &= tt_metal::CompileProgram(device, program);
        std::cout << "DONE COMPILING THE PROGRAM. GOING TO WRITE TO DRAM." << std::endl;
        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////

        auto activations = pack_bfloat16_vec_into_uint32_vec(src_vec);
        tt_metal::WriteToBuffer(src0_dram_buffer, activations);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tilized);
        tt_metal::WriteToBuffer(src1_dram_buffer, weights);
        std::cout << "DONE WRITING TO DEVICE. GOING TO CONFIGURE DEVICE WITH PROGRAM" << std::endl;
        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
        tt_metal::SetRuntimeArgs(
            program,
            generic_binary_reader_kernel,
            core,
            generic_binary_reader_args);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            writer_rt_args);
        tt_metal::WriteRuntimeArgsToDevice(device, program);
        std::cout << "DONE DEVICE CONFIGURE. GOING TO WRITE address map TO DEVICE L1" << std::endl;
        tt_metal::detail::WriteToDeviceL1(device, core, source_addresses_in_l1_addr, source_addresses);

        // DEBUG
        // Sanity check to verify address map in L1
        std::vector<uint32_t> source_addresses_in_l1;

        tt_metal::detail::ReadFromDeviceL1(device, core, source_addresses_in_l1_addr, source_addresses.size() * sizeof(uint32_t), source_addresses_in_l1);
        assert(source_addresses == source_addresses_in_l1);
        // END DEBUG

        std::cout << "DONE WRITING address map TO DEVICE L1. GOING TO LAUNCH KERNELS" << std::endl;
        pass &= tt_metal::LaunchKernels(device, program);
        std::cout << "DONE KERNELS. GOING TO READ FROM DRAM." << std::endl;

        std::vector<uint32_t> result_uint32;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_uint32);
        auto result_vec_tilized = unpack_uint32_vec_into_bfloat16_vec(result_uint32);
        assert(golden_act_matrix_tilized.size() == result_vec_tilized.size());
        auto result_vec = untilize(result_vec_tilized, act_rows, weight_cols);
        std::cout << "DONE READING FROM DRAM. GOING TO VALIDATE NOW." << std::endl;
        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        assert(golden_output_vec.size() == result_vec.size());
        pass &= (golden_output_vec == result_vec);
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
        log_fatal(LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
