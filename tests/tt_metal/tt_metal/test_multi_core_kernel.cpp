// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <array>
#include <tuple>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-logger/tt-logger.hpp>

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace {

std::tuple<Program, KernelHandle, KernelHandle> create_program(
    IDevice* /*device*/,
    uint32_t single_tile_size,
    const CoreRange& all_cores,
    const std::vector<uint32_t>& eltwise_unary_args) {
    Program program = CreateProgram();

    CoreCoord start_core = all_cores.start_coord;
    CoreCoord end_core = all_cores.end_coord;
    for (auto x = start_core.x; x <= end_core.x; x++) {
        for (auto y = start_core.y; y <= end_core.y; y++) {
            auto core = CoreCoord{x, y};
            uint32_t src0_cb_index = 0;
            uint32_t num_input_tiles = 8;
            CircularBufferConfig cb_src0_config =
                CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src0_cb_index, single_tile_size);
            CreateCircularBuffer(program, core, cb_src0_config);

            uint32_t ouput_cb_index = tt::CBIndex::c_16;
            uint32_t num_output_tiles = 1;
            CircularBufferConfig cb_output_config =
                CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(ouput_cb_index, single_tile_size);
            CreateCircularBuffer(program, core, cb_output_config);
        }
    }

    auto reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
        all_cores,
        ComputeConfig{.compile_args = eltwise_unary_args});

    return {std::move(program), reader_kernel, writer_kernel};
}

void set_rt_args(
    Program& program, KernelHandle kernel, const CoreRange& core_range, const std::array<uint32_t, 3>& rt_args) {
    for (auto x = core_range.start_coord.x; x <= core_range.end_coord.x; x++) {
        for (auto y = core_range.start_coord.y; y <= core_range.end_coord.y; y++) {
            CoreCoord core = CoreCoord(x, y);
            SetRuntimeArgs(program, kernel, core, rt_args);
        }
    }
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, MultiCoreKernelSameRuntimeArgs) {
    IDevice* dev = devices_[0]->get_devices()[0];

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {2, 2};
    CoreRange all_cores(start_core, end_core);

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    auto [program, reader_kernel_id, writer_kernel_id] =
        create_program(dev, single_tile_size, all_cores, compute_kernel_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    detail::WriteToBuffer(src_dram_buffer, src_vec);

    const std::array unary_reader_args{src_dram_buffer->address(), (std::uint32_t)0, (std::uint32_t)num_tiles};
    const std::array unary_writer_args{dst_dram_buffer->address(), (std::uint32_t)0, (std::uint32_t)num_tiles};

    set_rt_args(program, reader_kernel_id, all_cores, unary_reader_args);
    set_rt_args(program, writer_kernel_id, all_cores, unary_writer_args);

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(src_vec, result_vec);
}

TEST_F(MeshDeviceSingleCardFixture, MultiCoreKernelUniqueRuntimeArgs) {
    IDevice* dev = devices_[0]->get_devices()[0];

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {1, 1};
    CoreRange start_core_range(start_core, start_core);
    CoreRange core_group({0, 1}, {1, 1});
    CoreRange single_core({1, 0}, {1, 0});
    CoreRange all_cores(start_core, end_core);
    CoreRangeSet core_blocks = CoreRangeSet(std::vector{start_core_range, single_core, core_group});

    uint32_t single_tile_size = 2 * 1024;
    int32_t num_tiles = 2048;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};

    auto src_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer_1 = CreateBuffer(dram_config);
    auto dst_dram_buffer_2 = CreateBuffer(dram_config);
    auto dst_dram_buffer_3 = CreateBuffer(dram_config);

    vector<uint32_t> compute_kernel_args = {uint(num_tiles)};

    auto [program, reader_kernel_id, writer_kernel_id] =
        create_program(dev, single_tile_size, all_cores, compute_kernel_args);

    std::vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        src_dram_buffer->size(), 100, std::chrono::system_clock::now().time_since_epoch().count());

    detail::WriteToBuffer(src_dram_buffer, src_vec);

    const std::array unary_reader_args{src_dram_buffer->address(), (std::uint32_t)0, (std::uint32_t)num_tiles};
    const std::array unary_writer_args_1{dst_dram_buffer_1->address(), (std::uint32_t)0, (std::uint32_t)num_tiles};
    const std::array unary_writer_args_2{dst_dram_buffer_2->address(), (std::uint32_t)0, (std::uint32_t)num_tiles};
    const std::array unary_writer_args_3{dst_dram_buffer_3->address(), (std::uint32_t)0, (std::uint32_t)num_tiles};

    set_rt_args(program, reader_kernel_id, all_cores, unary_reader_args);
    int core_range_idx = 0;
    const std::array rt_args = {unary_writer_args_1, unary_writer_args_2, unary_writer_args_3};
    for (auto core_range : core_blocks.ranges()) {
        set_rt_args(program, writer_kernel_id, core_range, rt_args.at(core_range_idx++));
    }

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec_1;
    detail::ReadFromBuffer(dst_dram_buffer_1, result_vec_1);

    std::vector<uint32_t> result_vec_2;
    detail::ReadFromBuffer(dst_dram_buffer_2, result_vec_2);

    std::vector<uint32_t> result_vec_3;
    detail::ReadFromBuffer(dst_dram_buffer_3, result_vec_3);

    EXPECT_EQ(src_vec, result_vec_1);
    EXPECT_EQ(src_vec, result_vec_2);
    EXPECT_EQ(src_vec, result_vec_3);
}
