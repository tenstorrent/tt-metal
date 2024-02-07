// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>
#include <vector>
#include <string>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#include "common/bfloat16.hpp"
#include "constants.hpp"

#include "test_tiles.hpp"
#include "test_gold_impls.hpp"

using namespace tt;
using namespace constants;

using std::uint32_t;
using std::uint16_t;
using std::vector;
using std::string;

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: tests reduce kernel in HW dimensions (NCHW->NC11)
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    bool pass = true;
    bool multibank = true;

    for (int do_max = 0; do_max <= 1; do_max++) {
    log_info(LogTest, "Running reduce test for max={}", do_max);
    try {

        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device =
            tt_metal::CreateDevice(device_id);



        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};

        vector<uint32_t> shape = {1, 2, 7*TILE_HEIGHT, 5*TILE_WIDTH};
        //vector<uint32_t> shape = {1, 1, 1*TILE_HEIGHT, 1*TILE_WIDTH};
        //std::cout << "REDUCE HW SHAPE=" << shape[0] << " " << shape[1] << " " << shape[2] << " " << shape[3] << std::endl;
        uint32_t W = shape[3], H = shape[2], NC = shape[1]*shape[0];
        uint32_t HW = H*W;
        TT_FATAL(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
        TT_FATAL(H > 0 && W > 0 && NC > 0);
        uint32_t Wt = W/TILE_WIDTH;
        uint32_t Ht = H/TILE_HEIGHT;
        float scaler = 1.0f; // /(HW); // TODO(AP): this is broken
        uint32_t num_tensor_tiles = NC*H*W / (TILE_WIDTH*TILE_HEIGHT);

        uint32_t single_tile_bytes = 2 * 1024;
        uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t src_page_size = single_tile_bytes;
        uint32_t dst_page_size = single_tile_bytes;
        if (not multibank) {
            src_page_size = dram_buffer_bytes;
            dst_page_size = dram_buffer_bytes/(Ht*Wt);
        }

        tt_metal::InterleavedBufferConfig src_config{
                    .device=device,
                    .size = dram_buffer_bytes,
                    .page_size = src_page_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };
        tt_metal::InterleavedBufferConfig dst_config{
                    .device=device,
                    .size = dram_buffer_bytes/(Ht*Wt),
                    .page_size = dst_page_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

        auto src0_dram_buffer = CreateBuffer(src_config);
        uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
        auto dst_dram_buffer = CreateBuffer(dst_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t num_buffer_tiles = 32;
        // this buffer is used in transpose_hc.cpp NCRISC kernel
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_bytes);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_buffer_tiles = 32;
        // this buffer is used in writer_unary.cpp BRISC kernel
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_buffer_tiles * single_tile_bytes, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_bytes);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        tt_metal::CircularBufferConfig cb_scaler_reduce_tile_config = tt_metal::CircularBufferConfig(2 * single_tile_bytes, {{CB::c_in2, tt::DataFormat::Float16_b}})
            .set_page_size(CB::c_in2, single_tile_bytes);
        auto cb_scaler_reduce_tile = tt_metal::CreateCircularBuffer(program, core, cb_scaler_reduce_tile_config);

        TT_FATAL(num_tensor_tiles%Ht == 0);

        auto unary_reader_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank_reduce.cpp"
                      : "tt_metal/kernels/dataflow/reader_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            multibank ? "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp"
                      : "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(Ht),
            uint(Wt),
            uint(NC),
        };

        string compute_kernel;
        std::map<string, string> reduce_defines = {
            {"REDUCE_OP", do_max ? "PoolType::MAX" : "PoolType::SUM"},
            {"REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"}
        };
        auto reduce_hw_compute_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/reduce_hw.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = reduce_defines}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////



        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 1.0f, 0x1234, -0.48f);
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);



        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel,
            core,
            {
                dram_buffer_src0_addr,
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                num_tensor_tiles, 0, 0, 0, 0,
                *reinterpret_cast<uint32_t*>(&scaler),
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
                num_tensor_tiles/(Wt*Ht)
            }
        );



        tt_metal::detail::LaunchProgram(device, program);

        // The kernel will view the input as TILED32_4FACES
        vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);
        TT_FATAL(result_vec.size() == NC*32*32/2); // we are expecting one tile in H, and half the elements since the vector packs 2 uint16_ts

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////

        int argfail = -1;
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.06f;
            const float atol = 1e-2f;
            float maxabs = fmaxf(fabsf(a), fabsf(b));
            float absdiff = fabsf(a - b);
            auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
            if (!result)
                absdiff *= 1.0f; // breakpoint spot
            return result;
        };

        // recover a linear view of input vector for consumption by gold_ function
        auto u16_src0_vec = u16_from_u32_vector(src0_vec);
        vector<uint16_t> src_linear = convert_layout<uint16_t>(u16_src0_vec, shape, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
        vector<uint16_t> gold_reduced = gold_reduce_hw(src_linear, shape, scaler, do_max ? true : false); // result is uint16_t untilized

        // Tilize from row major and convert to pairs (uint32_t)
        vector<uint32_t> shapeR{shape[0], shape[1], 32, 32};
        auto gold_4f_u32 = u32_from_u16_vector(convert_layout<uint16_t>(gold_reduced, shapeR, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES));

        pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
        if (!pass)
            log_error(LogTest, "Failure position={}", argfail);

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
        TT_THROW("Test Failed");
    }
    }

    TT_FATAL(pass);

    return 0;
}
