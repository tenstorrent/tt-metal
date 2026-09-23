// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <vector>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"

namespace {

using tt::tt_metal::GenericMeshDeviceFixture;

class WaveletSfpuTest : public GenericMeshDeviceFixture {};

TEST_F(WaveletSfpuTest, HorizontalStencilRotateGoldenVector) {
    if (mesh_device_->arch() != tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "The wavelet SFPU rotate golden vector is validated on Wormhole B0";
    }

    constexpr uint32_t tile_side = 32;
    constexpr uint32_t tile_elements = tile_side * tile_side;
    constexpr uint32_t tile_bytes = tile_elements * sizeof(float);
    constexpr tt::CBIndex input_cb = tt::CBIndex::c_0;
    constexpr tt::CBIndex output_cb = tt::CBIndex::c_16;
    const CoreCoord core{0, 0};

    std::vector<uint32_t> a(tile_elements);
    std::vector<uint32_t> b(tile_elements);
    std::vector<uint32_t> expected(tile_elements);
    for (uint32_t row = 0; row < tile_side; ++row) {
        for (uint32_t column = 0; column < tile_side; ++column) {
            const uint32_t index = row * tile_side + column;
            a[index] = std::bit_cast<uint32_t>(static_cast<float>(index));
            b[index] = std::bit_cast<uint32_t>(static_cast<float>(tile_elements + index));
        }
    }
    for (uint32_t row = 0; row < tile_side; ++row) {
        for (uint32_t column = 0; column < tile_side; ++column) {
            const uint32_t index = row * tile_side + column;
            const uint32_t face_column = (column / 16) * 16;
            const uint32_t parity = column % 2;
            const uint32_t lane = (column % 16) / 2;
            const uint32_t source_column = face_column + parity + (lane == 0 ? 14 : 2 * (lane - 1));
            expected[index] = lane == 0 ? a[row * tile_side + source_column] : b[row * tile_side + source_column];
        }
    }

    auto a_tiled = tilize_swizzled(a, tile_side, tile_side);
    auto b_tiled = tilize_swizzled(b, tile_side, tile_side);
    const auto expected_tiled = tilize_swizzled(expected, tile_side, tile_side);
    a_tiled.insert(a_tiled.end(), b_tiled.begin(), b_tiled.end());

    auto& cq = mesh_device_->mesh_command_queue();
    const auto input_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = 2 * tile_bytes},
        {.page_size = tile_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device_.get());
    const auto output_buffer = tt::tt_metal::distributed::MeshBuffer::create(
        tt::tt_metal::distributed::ReplicatedBufferConfig{.size = tile_bytes},
        {.page_size = tile_bytes, .buffer_type = tt::tt_metal::BufferType::DRAM},
        mesh_device_.get());

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    tt::tt_metal::CreateCircularBuffer(
        program,
        core,
        tt::tt_metal::CircularBufferConfig(2 * tile_bytes, {{input_cb, tt::DataFormat::Float32}})
            .set_page_size(input_cb, tile_bytes));
    tt::tt_metal::CreateCircularBuffer(
        program,
        core,
        tt::tt_metal::CircularBufferConfig(tile_bytes, {{output_cb, tt::DataFormat::Float32}})
            .set_page_size(output_cb, tile_bytes));

    std::vector<uint32_t> reader_compile_args;
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_args);
    const auto reader = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_args});
    std::vector<uint32_t> writer_compile_args = {static_cast<uint32_t>(output_cb)};
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_args);
    const auto writer = tt::tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = writer_compile_args});
    tt::tt_metal::CreateKernel(
        program,
        "tests/ttnn/unit_tests/gtests/kernels/wavelet_horizontal_rotate.cpp",
        core,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = true});

    tt::tt_metal::SetRuntimeArgs(program, reader, core, {input_buffer->address(), 0, 0, 2});
    tt::tt_metal::SetRuntimeArgs(program, writer, core, {output_buffer->address(), 0, 1});

    tt::tt_metal::distributed::MeshWorkload workload;
    const auto zero = tt::tt_metal::distributed::MeshCoordinate(0, 0);
    const auto device_range = tt::tt_metal::distributed::MeshCoordinateRange(zero, zero);
    workload.add_program(device_range, std::move(program));
    tt::tt_metal::distributed::WriteShard(cq, input_buffer, a_tiled, zero);
    std::vector<uint32_t> output_initial(tile_elements, 0);
    tt::tt_metal::distributed::WriteShard(cq, output_buffer, output_initial, zero);
    tt::tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt::tt_metal::distributed::Finish(cq);

    std::vector<uint32_t> actual;
    tt::tt_metal::distributed::ReadShard(cq, actual, output_buffer, zero);
    EXPECT_EQ(actual, expected_tiled);
}

}  // namespace
