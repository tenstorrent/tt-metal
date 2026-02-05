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

TEST_F(MeshDispatchFixture, DataflowCb) {
    auto mesh_device = devices_[0];
    IDevice* dev = mesh_device->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1800;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    const uint32_t start_cb = 0;
    const uint32_t stride = 8;
    const uint32_t num_cbs_stride = (max_cbs_ / stride);
    const uint32_t topmost_cb = max_cbs_ - 1;

    const uint32_t total_cbs = num_cbs_stride + 1;
    ASSERT_EQ(num_tiles % total_cbs, 0) << "num_tiles must be divisible by total CBs";
    uint32_t num_tiles_per_cb = num_tiles / total_cbs;
    uint32_t num_cb_tiles = 8;

    auto create_cb = [&](uint32_t cb_index) {
        CircularBufferConfig cb_config =
            CircularBufferConfig(num_cb_tiles * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_index, single_tile_size);
        CreateCircularBuffer(program, core, cb_config);
    };

    // Create CBs at 0, 8, 16, ...
    for (uint32_t cb_idx = 0; cb_idx < max_cbs_; cb_idx += 8) {
        create_cb(cb_idx);
    }
    // Also test topmost
    create_cb(topmost_cb);

    std::vector<uint32_t> reader_cb_kernel_args = {start_cb, num_cbs_stride, stride, topmost_cb, 2};
    std::vector<uint32_t> writer_cb_kernel_args = {start_cb, num_cbs_stride, stride, topmost_cb, 4};

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
