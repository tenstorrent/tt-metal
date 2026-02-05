// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "mesh_dispatch_fixture.hpp"

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

// Tests dataflow through CBs at indices 0, 8, 16, ... and topmost CB
// Validates data integrity: DRAM -> Reader -> CB -> Writer -> DRAM (src == dst)
TEST_F(MeshDispatchFixture, DataflowCb) {
    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    // Tile configuration
    constexpr uint32_t single_tile_size = 2 * 1024;
    constexpr uint32_t cb_capacity_tiles = 8;
    constexpr uint32_t tiles_to_transfer_per_cb = 200;
    constexpr uint32_t reader_ublock_size = 2;
    constexpr uint32_t writer_ublock_size = 4;

    static_assert(
        tiles_to_transfer_per_cb % writer_ublock_size == 0,
        "tiles_to_transfer_per_cb must be divisible by writer_ublock_size");

    // CB index configuration
    constexpr uint32_t start_cb = 0;
    constexpr uint32_t stride = 8;
    const uint32_t strided_cb_count = max_cbs_ / stride;  // CBs at 0, 8, 16, ...
    const uint32_t topmost_cb = max_cbs_ - 1;
    const uint32_t total_cbs = strided_cb_count + 1;  // Strided + topmost

    const uint32_t num_tiles = tiles_to_transfer_per_cb * total_cbs;
    const uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    auto create_cb = [&](uint32_t cb_index) {
        CircularBufferConfig cb_config =
            CircularBufferConfig(cb_capacity_tiles * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_config);
    };

    // Create CBs at 0, 8, 16, ...
    for (uint32_t cb_idx = 0; cb_idx < max_cbs_; cb_idx += stride) {
        create_cb(cb_idx);
    }
    // Also test topmost
    create_cb(topmost_cb);

    std::vector<uint32_t> reader_cb_compile_args = {start_cb, strided_cb_count, stride, topmost_cb, reader_ublock_size};
    std::vector<uint32_t> writer_cb_compile_args = {start_cb, strided_cb_count, stride, topmost_cb, writer_ublock_size};

    auto reader_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_cb_compile_args});

    auto writer_cb_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_cb_test.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_cb_compile_args});

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
    detail::WriteToBuffer(src_dram_buffer, src_vec);

    SetRuntimeArgs(program, reader_cb_kernel, core, {dram_buffer_src_addr, 0, tiles_to_transfer_per_cb});
    SetRuntimeArgs(program, writer_cb_kernel, core, {dram_buffer_dst_addr, 0, tiles_to_transfer_per_cb});

    // Execute
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    this->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    // Validation
    EXPECT_EQ(src_vec, result_vec);
}
