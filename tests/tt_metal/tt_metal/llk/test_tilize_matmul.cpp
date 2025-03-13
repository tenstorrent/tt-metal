// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/tilization.hpp"
#include "test_golden_impls.hpp"

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::tilize_matmul {

std::vector<bfloat16> gold_matmul(
    std::vector<bfloat16>& in0, std::vector<bfloat16>& in1, const std::vector<uint32_t>& shape) {
    int R = shape.at(0);
    int C = shape.at(1);
    int I = shape.at(2);

    std::vector<bfloat16> golden(R * C);

    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            float res = 0;
            for (int k = 0; k < I; k++) {
                res += in0[i * I + k].to_float() * in1[k * C + j].to_float();
            }
            golden[i * C + j] = res;
        }
    }

    return golden;
}

void run_single_core_tilize_matmul(tt_metal::IDevice* device) {
    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t in0_cb_index = 0;
    constexpr uint32_t in1_cb_index = 1;
    constexpr uint32_t imm_cb_index = 2;
    constexpr uint32_t out_cb_index = 16;

    constexpr uint32_t single_tile_size = tile_width * tile_height * bfloat16::SIZEOF;

    uint32_t dram_buffer_size = single_tile_size;

    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    auto in0_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_in0_addr = in0_dram_buffer->address();
    auto in1_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_in1_addr = in1_dram_buffer->address();
    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_in0_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb_index, single_tile_size);
    auto l1_in0_cb = tt_metal::CreateCircularBuffer(program, core, l1_in0_cb_config);

    tt_metal::CircularBufferConfig l1_in1_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb_index, single_tile_size);
    auto l1_in1_cb = tt_metal::CreateCircularBuffer(program, core, l1_in1_cb_config);

    tt_metal::CircularBufferConfig l1_imm_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{imm_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(imm_cb_index, single_tile_size);
    auto l1_imm_cb = tt_metal::CreateCircularBuffer(program, core, l1_imm_cb_config);

    tt_metal::CircularBufferConfig l1_scr_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{3, tt::DataFormat::Float16_b}})
            .set_page_size(3, single_tile_size);
    auto l1_scr_cb = tt_metal::CreateCircularBuffer(program, core, l1_scr_cb_config);

    tt_metal::CircularBufferConfig l1_dst_cb_config =
        tt_metal::CircularBufferConfig(dram_buffer_size, {{out_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(out_cb_index, single_tile_size);
    auto l1_dst_cb = tt_metal::CreateCircularBuffer(program, core, l1_dst_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    auto compute_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/tilize_matmul.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .compile_args = {in0_cb_index, in1_cb_index, imm_cb_index, out_cb_index}});

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)dram_buffer_in0_addr,
            (uint32_t)0,  // in_0 dram bank id
            (uint32_t)dram_buffer_in1_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)dram_buffer_dst_addr,
            (uint32_t)0,
            (uint32_t)1,  // num_tiles
        });

    std::vector<bfloat16> in0 = generate_uniform_random_vector<bfloat16>(
        0.0f, 10.0f, dram_buffer_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<bfloat16> in1 = generate_uniform_random_vector<bfloat16>(
        0.0f, 10.0f, dram_buffer_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<bfloat16> golden = gold_matmul(in0, in1, {32, 32, 32});
    auto golden_packed = pack_vector<uint32_t, bfloat16>(golden);

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            std::cout << std::setw(9) << golden[i * 32 + j].to_float() << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    auto in0_packed = pack_vector<uint32_t, bfloat16>(in0);
    auto in1_packed = pack_vector<uint32_t, bfloat16>(in1);

    // in0 is tilized inside the kernel
    auto in1_packed_tilized = ::unit_tests::compute::gold_standard_tilize(in1_packed, {1, 1});

    tt_metal::detail::WriteToBuffer(in0_dram_buffer, in0_packed);
    tt_metal::detail::WriteToBuffer(in1_dram_buffer, in1_packed_tilized);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_packed_tilized;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, output_packed_tilized);
    auto output_packed = ::unit_tests::compute::gold_standard_untilize(output_packed_tilized, {1, 1});

    auto output = unpack_vector<bfloat16, uint32_t>(output_packed);

    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            std::cout << std::setw(9) << output[i * 32 + j].to_float() << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::endl;

    bool result = is_close_packed_vectors<bfloat16, uint32_t>(
        output_packed, golden_packed, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.05f); });
    ASSERT_TRUE(result);
}
}  // namespace unit_tests::compute::tilize_matmul

TEST_F(DeviceFixture, TensixComputeTilizeMatmul) {
    unit_tests::compute::tilize_matmul::run_single_core_tilize_matmul(this->devices_.at(0));
}

}  // namespace tt::tt_metal
