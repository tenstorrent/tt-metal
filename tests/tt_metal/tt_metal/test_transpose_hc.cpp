// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"
#include "tt_metal/detail/tt_metal.hpp"

#include "test_tiles.hpp"

using namespace tt;

using std::uint32_t;
using std::uint16_t;


//////////////////////////////////////////////////////////////////////////////////////////
// Reference CPU implementation of transpose_HC
//////////////////////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: tests transpose kernel for HC dimensions
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;
    bool multibank = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

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

        //vector<uint32_t> shape = {1, 96, 32*4, 32*5};
        vector<uint32_t> shape = {2, 32*3, 32*5, 32*2};
        uint32_t num_tensor_tiles = shape.at(0) * shape.at(1) * shape.at(2) * shape.at(3) / (32*32);

        uint32_t single_tile_bytes = 2 * 1024;
        uint32_t dram_buffer_bytes = single_tile_bytes * num_tensor_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        uint32_t page_size = single_tile_bytes;
        if (not multibank) {
            page_size = dram_buffer_bytes;
        }

        tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = dram_buffer_bytes,
                    .page_size = page_size,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

        auto src0_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();
        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
        auto dst_dram_buffer = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        uint32_t src0_cb_index = 0;
        uint32_t num_buffer_tiles = 2;
        // this buffer is used in transpose_hc.cpp NCRISC kernel
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_bytes);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        // this buffer is used in writer_unary.cpp BRISC kernel
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_buffer_tiles * single_tile_bytes, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_bytes);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        uint32_t W = shape[3], H = shape[2], C = shape[1], N = shape[0];
        uint32_t HW = H*W;
        uint32_t CHW = C*H*W;

        auto reader_kernel = tt_metal::CreateKernel(
            program,
            multibank ?
                "tests/tt_metal/tt_metal/test_kernels/dataflow/transpose_hc_8bank.cpp" :
                "tests/tt_metal/tt_metal/test_kernels/dataflow/transpose_hc.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            multibank ?
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp" :
                "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        vector<uint32_t> compute_kernel_args = {
            uint(num_tensor_tiles)
        };

        auto blank_binary_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////



        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(dram_buffer_bytes, 100, 0x1234);
        auto src_4f_16 = u16_from_u32_vector(src0_vec);
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);



        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                dram_buffer_src0_addr,
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                W, H, C, HW, N, CHW
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
                num_tensor_tiles
            }
        );



        tt_metal::detail::LaunchProgram(device, program);

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        ////////////////////////////////////////////////////////////////////////////
        //                      Validation & Teardown
        ////////////////////////////////////////////////////////////////////////////
        int argfail = -1;
        auto comparison_function = [](float a, float b) {
            const float rtol = 0.001f;
            const float atol = 1e-3f;
            float maxabs = fmaxf(fabsf(a), fabsf(b));
            float absdiff = fabsf(a - b);
            auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
            if (!result)
                absdiff *= 1.0f; // breakpoint spot
            return result;
        };

        // recover a linear view of input vector for consumption by gold_ function
        vector<uint16_t> src_linear = convert_layout<uint16_t>(src_4f_16, shape, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
        vector<uint16_t> gold_reduced = gold_transpose_hc(src_linear, shape); // result is uint16_t untilized

        // Tilize from row major and convert to pairs (uint32_t)
        vector<uint32_t> shapeR{shape[0], shape[2], shape[1], shape[3]};
        auto gold_16_4f = convert_layout<uint16_t>(gold_reduced, shapeR, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES);
        auto gold_4f_u32 = u32_from_u16_vector(gold_16_4f);
        auto u16_result = u16_from_u32_vector(result_vec);

        pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
        if (!pass)
            log_error(LogTest, "Failure position={}", argfail);

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
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);

    return 0;
}
