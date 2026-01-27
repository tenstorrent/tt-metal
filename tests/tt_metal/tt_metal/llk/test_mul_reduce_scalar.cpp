// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include "test_golden_impls.hpp"
#include "tt_metal/impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace mul_reduce_scalar {

namespace {

void run_mul_reduce_scalar_test(IDevice* device, uint32_t num_tiles = 1, float b_value = 1.0f) {
    uint32_t single_tile_size = 2 * 1024;

    log_info(LogTest, "Testing multiply + reduce scalar with {} tiles", num_tiles);
    log_info(LogTest, "Using random values for A and 1.0 for all B values");
    log_info(LogTest, "Expected: REDUCE_SCALAR sums all A[i] * 1.0 = sum of all A[i] across all tiles");

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    auto create_buffer_config = [&device](uint32_t size, uint32_t page_size) {
        return tt_metal::InterleavedBufferConfig{
            .device = device, .size = size, .page_size = page_size, .buffer_type = tt_metal::BufferType::DRAM};
    };

    uint32_t input_buffer_size = num_tiles * single_tile_size;
    auto src0_dram_buffer = CreateBuffer(create_buffer_config(input_buffer_size, single_tile_size));
    auto src1_dram_buffer = CreateBuffer(create_buffer_config(input_buffer_size, single_tile_size));
    auto dst_dram_buffer = CreateBuffer(create_buffer_config(single_tile_size, single_tile_size));

    uint32_t cb_tiles = std::max(8u, num_tiles);
    auto create_cb_config = [&single_tile_size](uint32_t cb_index, uint32_t num_tiles) {
        return tt_metal::CircularBufferConfig(num_tiles * single_tile_size, {{cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(cb_index, single_tile_size);
    };

    tt_metal::CreateCircularBuffer(program, core, create_cb_config(tt::CBIndex::c_0, cb_tiles));
    tt_metal::CreateCircularBuffer(program, core, create_cb_config(tt::CBIndex::c_1, cb_tiles));
    tt_metal::CreateCircularBuffer(program, core, create_cb_config(tt::CBIndex::c_16, cb_tiles));

    // Reader kernel - reads tiles for both input tensors
    auto dual_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_multiple_tiles.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    // Writer kernel - writes 1 output tile
    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    // Compute kernel - performs multiply + reduce scalar operation
    std::map<std::string, std::string> compute_defines = {{"REDUCE_OP", "PoolType::SUM"}};
    auto mul_reduce_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/mul_reduce_scalar.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,  // Highest precision (4x slower)
            .compile_args = {},
            .defines = compute_defines});

    // Reader kernel args: src0_addr, src0_bank_id, src1_addr, src1_bank_id, num_tiles
    const std::array<uint32_t, 5> reader_args = {
        src0_dram_buffer->address(),  // src0_addr
        0,                            // src0_bank_id
        src1_dram_buffer->address(),  // src1_addr
        0,                            // src1_bank_id
        num_tiles                     // num_tiles
    };

    // Writer kernel args: dst_addr, dst_bank, num_tiles
    const std::array<uint32_t, 3> writer_args = {
        dst_dram_buffer->address(),  // dst_addr
        0,                           // dst_bank
        1                            // num_tiles (1 tile output)
    };

    SetRuntimeArgs(program, dual_reader_kernel, core, reader_args);
    SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);

    // Compute kernel runtime args: num_tiles
    SetRuntimeArgs(
        program,
        mul_reduce_kernel,
        core,
        {
            num_tiles  // num_tiles
        });

    // Generate test data
    // Use fixed seed for reproducibility
    uint32_t seed = 12345;
    uint32_t byte_size = num_tiles * single_tile_size;

    // A: random values, B: all 1.0 for easier debugging (result = sum of A)
    std::vector<uint32_t> packed_input0 =
        generate_packed_uniform_random_vector<uint32_t, bfloat16>(0, 1.0f, byte_size / sizeof(bfloat16), seed);

    // Create B with all 1.0 values
    bfloat16 one_bf16 = bfloat16(1.0f);
    uint16_t one_u16 = std::bit_cast<uint16_t>(one_bf16);
    std::vector<uint32_t> packed_input1(byte_size / sizeof(uint32_t));
    for (size_t i = 0; i < packed_input1.size(); ++i) {
        // Pack two bfloat16 values (both 1.0) into one uint32_t
        packed_input1[i] = (static_cast<uint32_t>(one_u16) << 16) | one_u16;
    }

    std::vector<uint32_t> src0_vec = packed_input0;
    std::vector<uint32_t> src1_vec = packed_input1;

    tt_metal::detail::WriteToBuffer(*src0_dram_buffer, src0_vec);
    tt_metal::detail::WriteToBuffer(*src1_dram_buffer, src1_vec);
    tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(*dst_dram_buffer, result_vec);

    auto u16_src0_vec = u16_from_u32_vector(src0_vec);
    auto u16_src1_vec = u16_from_u32_vector(src1_vec);

    // Compute golden reference for REDUCE_SCALAR across ALL tiles
    float golden_scalar = 0.0f;
    // Process ALL tiles' worth of data
    for (size_t i = 0; i < u16_src0_vec.size(); ++i) {
        float val0 = static_cast<float>(std::bit_cast<bfloat16>(u16_src0_vec[i]));
        float val1 = static_cast<float>(std::bit_cast<bfloat16>(u16_src1_vec[i]));
        golden_scalar += val0 * val1;
    }

    auto u16_result_vec = u16_from_u32_vector(result_vec);

    // For REDUCE_SCALAR, the result is stored in the first element of the output tile
    float device_scalar = static_cast<float>(std::bit_cast<bfloat16>(u16_result_vec[0]));

    log_info(LogTest, "Golden scalar: {}", golden_scalar);
    log_info(LogTest, "Device scalar: {}", device_scalar);
    log_info(LogTest, "Difference: {}", std::abs(device_scalar - golden_scalar));
    log_info(
        LogTest, "Relative error: {:.2f}%", 100.0f * std::abs(device_scalar - golden_scalar) / std::abs(golden_scalar));

    // Use relative tolerance for random values
    float tolerance = std::max(0.01f * std::abs(golden_scalar), 0.1f);
    EXPECT_NEAR(device_scalar, golden_scalar, tolerance);
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalar1Tile) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 1, 1.0f);
}

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalar2Tiles) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 2, 1.0f);
}

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalar3Tiles) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 3, 1.0f);
}

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalar7Tiles) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 7, 1.0f);
}

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalar8Tiles) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 8, 1.0f);
}
}  // namespace mul_reduce_scalar
