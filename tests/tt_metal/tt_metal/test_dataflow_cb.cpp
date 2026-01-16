// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>

using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, DataflowCb) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    int num_cbs = 1;
    ASSERT_EQ(num_tiles % num_cbs, 0) << "num_tiles must be divisible by num_cbs";
    int num_tiles_per_cb = num_tiles / num_cbs;

    uint32_t cb0_index = 0;
    uint32_t num_cb_tiles = 8;
    CircularBufferConfig cb0_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb0_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb0_index, single_tile_size);
    CreateCircularBuffer(program, core, cb0_config);

    uint32_t cb1_index = 8;
    CircularBufferConfig cb1_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb1_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb1_index, single_tile_size);
    CreateCircularBuffer(program, core, cb1_config);

    uint32_t cb2_index = 16;
    CircularBufferConfig cb2_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb2_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb2_index, single_tile_size);
    CreateCircularBuffer(program, core, cb2_config);

    uint32_t cb3_index = 24;
    CircularBufferConfig cb3_config =
        CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb3_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb3_index, single_tile_size);
    CreateCircularBuffer(program, core, cb3_config);

    std::vector<uint32_t> reader_cb_kernel_args = {8, 2};
    std::vector<uint32_t> writer_cb_kernel_args = {8, 4};

    auto reader_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_cb_kernel_args});

    auto writer_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_cb_kernel_args});

    // Execute
    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(program, reader_cb_kernel, core, {dram_buffer_src_addr, 0, (uint32_t)num_tiles_per_cb});
    SetRuntimeArgs(program, writer_cb_kernel, core, {dram_buffer_dst_addr, 0, (uint32_t)num_tiles_per_cb});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validation
    EXPECT_EQ(src_vec, result_vec);
}
