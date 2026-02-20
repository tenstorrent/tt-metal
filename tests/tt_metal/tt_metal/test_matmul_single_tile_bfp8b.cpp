// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat8.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/bfloat_utils.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, MatmulSingleTileBfp8b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = tt::tile_size(tt::DataFormat::Bfp8_b);
    TT_FATAL(single_tile_size == (256 * 4) + (16 * 4), "Error");
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto mm_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {1, 1, 1, 1, 1, 1, 1};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    // Execute
    std::vector<uint32_t> activations = test_utils::create_random_vector_of_bfp8(
        dram_buffer_size,
        /*is_exp_a=*/false,
        100,
        std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src0_dram_buffer, activations);

    int num_float_in_tile = 32 * 32;
    std::vector<float> vec(num_float_in_tile, (float)0);
    for (int i = 0; i < 32; i++) {
        vec.at((i * 32) + i) = (float)1;
    }
    std::vector<uint32_t> weights =
        pack_as_bfp8_tiles(tt::stl::make_const_span(vec), /*row_major_input=*/true, /*is_exp_a=*/false);

    detail::WriteToBuffer(src1_dram_buffer, weights);

    SetRuntimeArgs(
        program,
        mm_reader_kernel,
        core,
        {src0_dram_buffer->address(),
         0,
         src1_dram_buffer->address(),
         0,
         1,
         1,
         1,
         1 * single_tile_size,
         1 * single_tile_size});

    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, num_tiles});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validation - matmul with identity should return same result
    EXPECT_EQ(activations, result_vec);
}
