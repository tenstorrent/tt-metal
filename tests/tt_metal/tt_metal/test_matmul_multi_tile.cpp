#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"
#include "tt_metal/llrt/test_libs/debug_mailbox.hpp"

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

// Given a tensor that is row-major datums, make it tilized
// so that its row major within a tile, and each tile's data
// is contiguous
template <typename T>
std::vector<T> tilize(std::vector<T> data, int rows, int cols) {
    TT_ASSERT(rows % 32 == 0);
    TT_ASSERT(cols % 32 == 0);
    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    std::vector<T> result;
    for(auto r = 0; r < num_tiles_r; r++) {
        for(auto c = 0; c < num_tiles_c; c++) {
            for(auto j = 0; j < 32; j++) { // tile rows
                for(auto i = 0; i < 32; i++) { // tile cols
                    // each row of tiles is 32x32 * num_tiles_c
                    // each row within the row of tiles is cols
                    // each col of tiles is 32
                    // pick row of tiles, pick the row within the tile, pick col tile
                    int index = r * 32 * 32 * num_tiles_c + j * cols + c * 32 + i;
                    result.push_back(data.at(index));
                }
            }
        }
    }
    return result;
}


// Given a tilized data (each tile's data is contiguous and row major within the tile)
// transform it back to row major full tensor. (This function inverts the tilize() function)
template <typename T>
std::vector<T> untilize(std::vector<T> data, int rows, int cols) {
    TT_ASSERT(rows % 32 == 0);
    TT_ASSERT(cols % 32 == 0);
    int num_tiles_r = rows / 32;
    int num_tiles_c = cols / 32;
    std::vector<T> result;
    for(auto r = 0; r < num_tiles_r; r++) {
        for(auto i = 0; i < 32; i++) {
            for(auto c = 0; c < num_tiles_c; c++) {
                int offset = r * 32 * 32 * num_tiles_c + c * 32 * 32 + i * 32;
                for(auto j = 0; j < 32; j++) {
                    result.push_back(data.at(offset + j));
                }
            }
        }
    }

    return result;
}

// Transpose 2D matrix of tiles so that its column major of tiles instead of row major.
// this is usually used for activation so that blocks data is contiguous in memory
// until we have a more generalized read kernel that can read tiles from different
// location in memory to make up a block in the activations CB
std::vector<std::uint32_t> transpose_tiles(std::vector<std::uint32_t> data, int row_tiles, int col_tiles) {
    std::vector<std::uint32_t> result;
    int tile_size = 512;
    for(int c = 0; c < col_tiles; c++) {
        for(int r = 0 ; r < row_tiles; r++) {
            int offset = tile_size * col_tiles * r + c * tile_size;
            for(int k = 0; k < tile_size; k++) {
                result.push_back(data.at(offset + k));
            }
        }
    }
    return result;
}

bool run_matmul(const tt::ARCH& arch, const bool with_bias) {
    bool pass = true;

    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);
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
        uint32_t M = 4;
        uint32_t K = 4;
        uint32_t N = K;
        TT_ASSERT(M * K * 32 * 32 <= (64 * 16 * 16));
        uint32_t single_tile_size = 2 * 1024;
        uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        auto src0_dram_buffer = tt_metal::Buffer(device, dram_buffer_size_act, dram_buffer_size_act, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = tt_metal::Buffer(device, dram_buffer_size_weights, dram_buffer_size_weights, tt_metal::BufferType::DRAM);

        tt_metal::Buffer src2_dram_buffer;
        if (with_bias) {
            src2_dram_buffer = tt_metal::Buffer(device, single_tile_size * N, single_tile_size * N, tt_metal::BufferType::DRAM);
        }

        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size_out, dram_buffer_size_out, tt_metal::BufferType::DRAM);

        auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();

        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 200 * 1024;
        uint32_t cb0_tiles = M * 2;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            src0_cb_index,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src0_cb_addr
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
        uint32_t cb1_tiles = N * 2;
        auto cb_src1 = tt_metal::CreateCircularBuffer(
            program,
            src1_cb_index,
            core,
            cb1_tiles,
            cb1_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            src1_cb_addr
        );

        if (with_bias) {
            uint32_t src2_cb_index = 2;
            uint32_t src2_cb_addr = 400 * 1024;
            uint32_t cb2_tiles = N * 2;
            auto cb_src2 = tt_metal::CreateCircularBuffer(
                program,
                src2_cb_index,
                core,
                cb2_tiles,
                cb2_tiles * single_tile_size,
                tt::DataFormat::Float16_b,
                src2_cb_addr
            );
        }

        uint32_t output_cb_addr = 500 * 1024;
        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t intermediate_cb_index = 24;

        // NOTE: intermediate and output CB share same address space since we operate it on it sequentially, not in parallel
        uint32_t num_output_tiles = M*N;
        CoreRangeSet cores(std::set<CoreRange>{CoreRange{.start=core, .end=core}});
        auto output_cb = tt_metal::CreateCircularBuffers(
            program,
            {ouput_cb_index, intermediate_cb_index},
            cores,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b,
            output_cb_addr
        );

        string reader_kernel = "tt_metal/kernels/dataflow/reader_matmul_with_bias_blocked.cpp";

        auto mm_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            reader_kernel,
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            1, // block_tile_dim, within block, how many tiles are on the K dim
            M, // dst_tile_rows
            N, // dst_tile_cols
            K, // block_cnt, across blocks, how many tiles are on the K dim
            M, // in0_block_tile_cnt, M * block_tile_dim
            N, // in1_block_tile_cnt,  N * block_tile_dim
            (M * N), // out_block_tile_cnt
            with_bias // whether or not to use bias
        };

        string compute_kernel_name;
        compute_kernel_name = "tt_metal/kernels/compute/matmul_with_bias.cpp";

        auto mm_kernel = tt_metal::CreateComputeKernel(
            program,
            compute_kernel_name,
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        pass &= tt_metal::CompileProgram(device, program);

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
        auto activations_tilized = tilize(tensor.get_values(), M * 32, K * 32);
        auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
        auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        auto activations_tile_transposed = transpose_tiles(activations, M, K);
        tt_metal::WriteToBuffer(src0_dram_buffer, activations_tile_transposed);

        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 32x32 identity
        auto identity_tilized = tilize(identity, K * 32, N * 32);
        auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        tt_metal::WriteToBuffer(src1_dram_buffer, weights);

        if (with_bias) {
            vector<uint32_t> bias(N * 512, 0); // Just a zero bias, since the output check is identity
            tt_metal::WriteToBuffer(src2_dram_buffer, bias);
        }

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        vector<uint32_t> reader_l1_args = {
            src0_dram_buffer.address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            src1_dram_buffer.address(),
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            K,
            M,
            N,
            M * single_tile_size,
            N * single_tile_size,
            with_bias
        };

        if (with_bias) {
            auto dram_src2_noc_xy = src2_dram_buffer.noc_coordinates();
            vector<uint32_t> bias_args = {
                src2_dram_buffer.address(),
                (std::uint32_t)dram_src2_noc_xy.x,
                (std::uint32_t)dram_src2_noc_xy.y,
                N,
                N * single_tile_size
            };

            for (uint32_t arg: bias_args) {
                reader_l1_args.push_back(arg);
            }
        }

        tt_metal::SetRuntimeArgs(
            program,
            mm_reader_kernel,
            core,
            reader_l1_args);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            {dst_dram_buffer.address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            M * N});

        CoreCoord debug_core = {1, 1};
        read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 0);

        tt_metal::WriteRuntimeArgsToDevice(device, program);
        pass &= tt_metal::LaunchKernels(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        auto result_flat_layout = convert_to_flat_layout(result_bfp16);
        auto result_untilized = untilize(result_flat_layout, M*32, N*32);

        pass &= (tensor.get_values() == result_untilized);

        DeallocateBuffer(src0_dram_buffer);
        DeallocateBuffer(src1_dram_buffer);
        if (with_bias) {
            DeallocateBuffer(src2_dram_buffer);
        }
        DeallocateBuffer(dst_dram_buffer);

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

    return pass;
}

int main(int argc, char **argv) {
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
    const tt::ARCH arch = tt::get_arch_from_string(arch_name);
    pass &= run_matmul(arch, false);
    pass &= run_matmul(arch, true);

    TT_ASSERT(pass);

    return 0;
}
