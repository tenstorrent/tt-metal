#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "tensor/tensor.hpp"
#include "test_tiles.hpp"
#include "llrt/tests/test_libs/debug_mailbox.hpp"
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

void print_vec(std::vector<bfloat16> data, int rows, int cols, string name) {
    std::cout<<name<<": "<<std::endl;
    int index = 0;
    for(int i = 0 ; i < rows ; i++) {
        for(int j = 0 ; j < cols; j++) {
            std::cout<<data.at(index).to_float()<<", ";
            index++;
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl;
}

void print_faces(std::vector<bfloat16> data, string name, string fname) {
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

bool test_matmul_large_block(bool activations_rm) {
    bool pass = true;

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Grayskull Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int pci_express_slot = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= tt_metal::InitializeDevice(device);;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program *program = new tt_metal::Program();

        tt_xy_pair core = {0, 0};
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

        uint32_t dram_buffer_src0_addr = 0;
        int dram_src0_channel_id = 0;
        uint32_t dram_buffer_src1_addr = 0;
        int dram_src1_channel_id = 1;
        uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        int dram_dst_channel_id = 0;

        auto src0_dram_buffer = tt_metal::CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_size_act, dram_buffer_src0_addr);
        auto src1_dram_buffer = tt_metal::CreateDramBuffer(device, dram_src1_channel_id, dram_buffer_size_weights, dram_buffer_src1_addr);
        auto dst_dram_buffer = tt_metal::CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size_out, dram_buffer_dst_addr);

        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t src0_cb_addr = 150 * 1024;
        uint32_t cb0_tiles = M * in0_block_w * 2;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t src0_tilized_index = 25;
        uint32_t src_0_tilized_cb_addr = 500 * 1024;
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_tilized_index,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            src_0_tilized_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t src1_cb_index = 1;
        uint32_t src1_cb_addr = 300 * 1024;
        uint32_t cb1_tiles = N * in0_block_w * 2;
        auto cb_src1 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src1_cb_index,
            core,
            cb1_tiles,
            cb1_tiles * single_tile_size,
            src1_cb_addr,
            tt::DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t output_cb_addr = 400 * 1024;
        uint32_t num_output_tiles = M * N;
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

        uint32_t interm0_cb_index = 24;
        uint32_t interm0_cb_addr = 400 * 1024;
        uint32_t interm0_cb_tiles = M * N;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            interm0_cb_tiles,
            interm0_cb_tiles * single_tile_size,
            interm0_cb_addr,
            tt::DataFormat::Float16_b
        );

        std::vector<uint32_t> mm_reader_rt_args{
            dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            dram_buffer_src1_addr,
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            (std::uint32_t)(K/in0_block_w), // num_blocks
            M * in0_block_w, // input 0 block num tiles
            N * in0_block_w, // input 1 block num tiles
            M * in0_block_w * single_tile_size, // input 0 block bytes
            N * in0_block_w * single_tile_size}; // input 1 block bytes

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

        auto mm_reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_matmul_blocked.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);

        auto unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unswizzle.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);

        int num_blocks = (K/in0_block_w);

        int in0_num_subblocks = (M/out_subblock_h);
        int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
        int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

        int in1_num_subblocks = (N/out_subblock_w);
        int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
        int in1_per_core_w = out_subblock_w * in1_num_subblocks;

        int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

        int in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;

        TT_ASSERT(in0_subblock_h * in0_block_w * in0_num_subblocks == in0_block_num_tiles);
        TT_ASSERT(in0_block_w == K);

        vector<uint32_t> compute_kernel_args;
        if (activations_rm) {

            compute_kernel_args = {
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
            };
        } else {

            compute_kernel_args = {
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
                uint(out_subblock_num_tiles),
            };
        }

        vector<string> arg_names = {
            "in0_block_w", "in0_num_subblocks", "in0_block_num_tiles", "in0_subblock_num_tiles", "in0_subblock_h",
            "in1_num_subblocks", "in1_block_num_tiles", "in1_per_core_w",
            "num_blocks",
            "out_subblock_h", "out_subblock_w", "out_subblock_num_tiles",
        };

        int i = 0;
        std::cout << "Args: " << std::endl;
        for (uint32_t v: compute_kernel_args) {
            std::cout << arg_names.at(i++) << ": " << v << std::endl;
        }
        std::cout << std::endl;

        tt_metal::ComputeKernelArgs *mm_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;

        string compute_kernel;
        if (activations_rm) {
            compute_kernel = "kernels/compute/3T/matmul_large_block_zm_activations_rm";
        } else {
            compute_kernel = "kernels/compute/matmul_large_block_zm.cpp";
        }

        auto mm_kernel = tt_metal::CreateComputeKernel(
            program,
            compute_kernel,
            core,
            mm_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////
        bool skip_hlkc = false;
        if (activations_rm) {
            pass &= tt_metal::CompileProgramNew(device, program);
        } else {
            pass &= tt_metal::CompileProgram(device, program, skip_hlkc);
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE shape = {1, 1, M * 32, K * 32};
        tt::Tensor<bfloat16> tensor = tt::initialize_tensor<bfloat16>(shape, tt::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());

        vector<uint32_t> activations;
        if (activations_rm) {
            activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
        } else {
            auto activations_tilized = tilize(tensor.get_values(), M * 32, K * 32);
            auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
            activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
        }
        pass &= tt_metal::WriteToDeviceDRAM(src0_dram_buffer, activations);

        auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 32x32 identity
        auto identity_tilized = tilize(identity, K * 32, N * 32);
        auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
        auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
        pass &= tt_metal::WriteToDeviceDRAM(src1_dram_buffer, weights);

        pass &= tt_metal::ConfigureDeviceWithProgram(device, program);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            mm_reader_kernel,
            core,
            mm_reader_rt_args);

        tt_metal::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            writer_rt_args);

        tt_xy_pair debug_core = {1, 1};
        // read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 0);
        // read_trisc_debug_mailbox(device->cluster(), 0, debug_core, 1);

        pass &= tt_metal::LaunchKernels(device, program);
        std::vector<uint32_t> result_vec;
        tt_metal::ReadFromDeviceDRAM(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
        auto result_flat_layout = convert_to_flat_layout(result_bfp16);
        auto result_untilized = untilize(result_flat_layout, M*32, N*32);
        pass &= (tensor.get_values() == result_untilized);

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
    pass &= test_matmul_large_block(true);
    pass &= test_matmul_large_block(false);
    TT_ASSERT(pass);
}
