// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <umd/device/types/core_coordinates.hpp>

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

TEST_F(MeshDeviceSingleCardFixture, MatmulSingleTileOutputInL1) {
    IDevice* dev = devices_[0]->get_devices()[0];
    Program program = CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles;

    InterleavedBufferConfig dram_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::DRAM};
    InterleavedBufferConfig l1_config{
        .device = dev, .size = dram_buffer_size, .page_size = dram_buffer_size, .buffer_type = BufferType::L1};

    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_l1_buffer = CreateBuffer(l1_config);

    auto l1_dst_noc_xy = dev->virtual_core_from_logical_core(
        dst_l1_buffer->allocator()->get_logical_core_from_bank_id(0), CoreType::WORKER);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
    CreateCircularBuffer(program, core, cb_output_config);

    auto mm_reader_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto unary_writer_kernel = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_1.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {1, 1, 1, 1, 1, 1, 1};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
        core,
        ComputeConfig{.compile_args = compute_kernel_args});

    // Execute
    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(
        shape, tt::deprecated::Initialize::RANDOM, 0, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tile_layout =
        convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(tensor.get_values()));
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    detail::WriteToBuffer(src0_dram_buffer, activations);

    auto identity = create_identity_matrix(32, 32, 32);
    auto weights_tile_layout = convert_layout_tile_swizzled_to_tile_nfaces(tt::stl::make_const_span(identity));
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
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

    SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {dst_l1_buffer->address(), (std::uint32_t)l1_dst_noc_xy.x, (std::uint32_t)l1_dst_noc_xy.y, num_tiles});

    detail::LaunchProgram(dev, program);

    std::vector<uint32_t> result_vec;
    detail::ReadFromBuffer(dst_l1_buffer, result_vec);

    // Validation
    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    auto result_flat_layout = convert_layout_tile_nfaces_to_tile_swizzled(tt::stl::make_const_span(result_bfp16));
    EXPECT_EQ(tensor.get_values(), result_flat_layout);
}
