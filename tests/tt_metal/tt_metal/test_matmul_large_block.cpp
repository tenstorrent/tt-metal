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

void print_faces(std::vector<bfloat16> data, string name) {
    std::cout<<name<<": "<<std::endl;
    int index = 0;

    int tile_index = 0;
    int face_index = 0;
    for(int i = 0; i < data.size(); i++) {
        if(i % 256 == 0 ){
            std::cout<<"Tile "<<tile_index / 4<<std::endl;
            std::cout<<"Face = "<<face_index<<std::endl;
            face_index++;
            tile_index++;
            if(face_index == 4) {
                face_index = 0;
            }
        }
        std::cout<<data.at(i).to_float()<<", ";
        if( (i+1) % 16 == 0) {
            std::cout<<std::endl;
        }
    }
    std::cout<<std::endl;
}

void create_CBs_for_fused_matmul(tt_metal::Program &program, tt_metal::Device* device, CoreCoord core, bool activations_rm, bool output_rm, uint32_t M, uint32_t N, uint32_t in0_block_w, uint32_t out_subblock_h) {

    uint32_t num_bytes_for_df = 2;

    uint32_t in0_cb                                   = 0;
    uint32_t in1_cb                                   = 1;
    uint32_t tilize_mode_tilized_in0_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;
    CoreRangeSet cores(std::set<CoreRange>{CoreRange{.start=core, .end=core}});

    // Invariants
    uint32_t cb0_tiles = M * in0_block_w * 2;
    auto cb_in0 = tt_metal::CreateCircularBuffer(
        program,
        in0_cb,
        core,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    uint32_t cb1_tiles = N * in0_block_w * 2;
    auto cb_in1 = tt_metal::CreateCircularBuffer(
        program,
        in1_cb,
        core,
        cb1_tiles,
        cb1_tiles * single_tile_size,
        tt::DataFormat::Float16_b
    );

    if (not activations_rm and not output_rm) { // no tilize, no untilize
        // Partials share same L1 address space as output
        auto cb_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            {matmul_partials_cb, out0_cb},
            cores,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    } else if (not activations_rm and output_rm) { // no tilize, just untilize

        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );


        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );


    } else if (activations_rm and not output_rm) { // just tilize, no untilize

        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            tilize_mode_tilized_in0_cb,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_matmul_partials = tt_metal::CreateCircularBuffers(
            program,
            {matmul_partials_cb, out0_cb},
            cores,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    } else { // tilize activations and untilize output

        // Used for placing tilized activations
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            tilize_mode_tilized_in0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Shares same address space as matmul partials
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_final_matmul_partials_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N; // Only space for one row
        auto cb_reblock = tt_metal::CreateCircularBuffer(
            program,
            untilize_mode_reblock_cb,
            core,
            reblock_cb_tiles,
            reblock_cb_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            out0_cb,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
    }
}

bool test_matmul_large_block(const tt::ARCH& arch, bool activations_rm, bool output_rm) {
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
        uint32_t M = 8;
        uint32_t K = 4;
        uint32_t N = K;
        int out_subblock_h = 4;
        int out_subblock_w = 2;
        int in0_block_w = K;

        uint32_t single_tile_size = 2 * 1024;
        TT_ASSERT(M * in0_block_w * single_tile_size * 2 <= 150*1024);
        TT_ASSERT(N * in0_block_w * single_tile_size * 2 <= 100*1024);
        TT_ASSERT(M * N * single_tile_size <= 600*1024);
        uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        auto src0_dram_buffer = tt_metal::Buffer(device, dram_buffer_size_act, dram_buffer_size_act, tt_metal::BufferType::DRAM);
        auto src1_dram_buffer = tt_metal::Buffer(device, dram_buffer_size_weights, dram_buffer_size_weights, tt_metal::BufferType::DRAM);
        auto dst_dram_buffer = tt_metal::Buffer(device, dram_buffer_size_out, dram_buffer_size_out, tt_metal::BufferType::DRAM);

        auto dram_src0_noc_xy = src0_dram_buffer.noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer.noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer.noc_coordinates();

        std::vector<uint32_t> mm_reader_rt_args{
            src0_dram_buffer.address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            src1_dram_buffer.address(),
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            (std::uint32_t)(K/in0_block_w), // num_blocks
            M * in0_block_w, // input 0 block num tiles
            N * in0_block_w, // input 1 block num tiles
            M * in0_block_w * single_tile_size, // input 0 block bytes
            N * in0_block_w * single_tile_size}; // input 1 block bytes


        std::vector<uint32_t> writer_rt_args;
        string writer_kernel;
        if (output_rm) {
            writer_kernel = "tt_metal/kernels/dataflow/writer_unary.cpp";
            writer_rt_args = {
                dst_dram_buffer.address(),
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                uint(M * N)
            };
        } else {
            writer_kernel = "tt_metal/kernels/dataflow/writer_unswizzle.cpp";
            writer_rt_args = {
                dst_dram_buffer.address(),
                (std::uint32_t)dram_dst_noc_xy.x,
                (std::uint32_t)dram_dst_noc_xy.y,
                (std::uint32_t)out_subblock_h, // num tiles per sub block m
                (std::uint32_t)out_subblock_w, // num tiles per sub block n
                (std::uint32_t)M/out_subblock_h, // num sub blocks m
                (std::uint32_t)N/out_subblock_w, // num sub blocks n
                (std::uint32_t)out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row within sub-block
                (std::uint32_t)out_subblock_h * out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row of sub-blocks
                (std::uint32_t)out_subblock_w*single_tile_size
            }; // bytes offset to next sub-block
        }

        auto mm_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            writer_kernel,
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        int num_blocks = (K/in0_block_w);

        int in0_num_subblocks = (M/out_subblock_h);
        int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
        int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        int in1_num_subblocks = (N/out_subblock_w);
        int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
        int in1_per_core_w = out_subblock_w * in1_num_subblocks;

        int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

        int in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;

        create_CBs_for_fused_matmul(program, device, core, activations_rm, output_rm, M, N, in0_block_w, out_subblock_h);

        TT_ASSERT(in0_subblock_h * in0_block_w * in0_num_subblocks == in0_block_num_tiles);
        TT_ASSERT(in0_block_w == K);

        vector<uint32_t> compute_kernel_args = {
            uint(in0_block_w),
            uint(in0_num_subblocks),
            uint(in0_block_num_tiles),
            uint(in0_subblock_num_tiles),
            uint(in0_subblock_h),

            uint(in1_num_subblocks),
            uint(in1_block_num_tiles),
            uint(in1_per_core_w),

            uint(num_blocks),

            uint(out_subblock_h),
            uint(out_subblock_w),
            uint(out_subblock_num_tiles),

            uint(activations_rm),
            uint(output_rm)
        };

        string compute_kernel = "tt_metal/kernels/compute/matmul_large_block.cpp";

        auto mm_kernel = tt_metal::CreateComputeKernel(
            program,
            compute_kernel,
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

        vector<uint32_t> activations;
        if (activations_rm) {
            activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        } else {
            auto activations_tilized = tilize(tensor.get_values(), M * 32, K * 32);
            auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
            activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        }
        tt_metal::WriteToBuffer(src0_dram_buffer, activations);

        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 32x32 identity
        auto identity_tilized = tilize(identity, K * 32, N * 32);
        auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        tt_metal::WriteToBuffer(src1_dram_buffer, weights);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::SetRuntimeArgs(
            program,
            mm_reader_kernel,
            core,
            mm_reader_rt_args);

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel,
            core,
            writer_rt_args);

        CoreCoord debug_core = {1, 1};

        tt_metal::WriteRuntimeArgsToDevice(device, program);

        read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 0);
        pass &= tt_metal::LaunchKernels(device, program);
        // read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 1);
        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        if (output_rm) {
            pass &= (tensor.get_values() == result_bfp16);
            if (not pass) {
                print_faces(result_bfp16, "Result");
            }
        } else {
            auto result_flat_layout = convert_to_flat_layout(result_bfp16);
            auto result_untilized = untilize(result_flat_layout, M*32, N*32);
            pass &= (tensor.get_values() == result_untilized);
            if (not pass) {
                print_faces(result_untilized, "Result");
            }
        }

        if (not pass) {
            print_faces(tensor.get_values(), "Golden");
        }

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

    // Row major input, tilized output
    pass &= test_matmul_large_block(arch, true, false);

    // Row major input, untilized output
    pass &= test_matmul_large_block(arch, true, true);

    // Tilized input, tilized output
    pass &= test_matmul_large_block(arch, false, false);

    // Tilized input, untilized output
    pass &= test_matmul_large_block(arch, false, true);

    TT_ASSERT(pass);
}
