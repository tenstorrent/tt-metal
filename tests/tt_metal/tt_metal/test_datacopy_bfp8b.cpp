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

TEST_F(MeshDeviceSingleCardFixture, DatacopyBfp8b) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = tt::tile_size(tt::DataFormat::Bfp8_b);
    TT_FATAL(single_tile_size == (256 * 4) + (16 * 4), "Error");
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Bfp8_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    // Execute
    std::vector<uint32_t> src_vec = test_utils::create_random_vector_of_bfp8(
        dram_buffer_size,
        /*is_exp_a=*/false,
        100,
        std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(program, unary_reader_kernel, core, {dram_buffer_src_addr, 0, num_tiles});
    SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, 0, num_tiles});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validation
    EXPECT_EQ(src_vec, result_vec);
}
