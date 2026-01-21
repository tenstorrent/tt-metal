// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

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

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace mul_reduce_scalar {

namespace {

void run_mul_reduce_scalar_test(IDevice* device, uint32_t num_tiles = 1, float b_value = 1.0f) {
    uint32_t single_tile_size = 2 * 1024;

    log_info(LogTest, "Testing multiply + reduce scalar");
    log_info(LogTest, "Input A: F0=0, F1=1, F2=2, F3=3 (256 elements per face)");
    log_info(LogTest, "Input B: all {}", b_value);
    log_info(LogTest, "Expected: REDUCE_SCALAR sums all A[i] * B[i] elements");
    log_info(LogTest, "  Total = {} x (256x0 + 256x1 + 256x2 + 256x3) = {}", b_value, b_value * 1536.0);

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

    uint32_t cb_tiles = 8;
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
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp",
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
        tt_metal::ComputeConfig{.compile_args = {}, .defines = compute_defines});

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

    // Generate data for num_tiles with deterministic pattern (but function will only process 1 tile)
    std::vector<uint32_t> src0_vec(num_tiles * single_tile_size / sizeof(uint32_t));
    constexpr uint32_t elements_per_face = 256;

    for (uint32_t tile = 0; tile < num_tiles; ++tile) {
        uint32_t tile_offset = tile * (single_tile_size / sizeof(uint32_t));
        for (uint32_t face = 0; face < 4; ++face) {
            float face_value = static_cast<float>(face);
            for (uint32_t elem = 0; elem < elements_per_face / 2; ++elem) {
                bfloat16 bf_val(face_value);
                src0_vec[tile_offset + face * (elements_per_face / 2) + elem] =
                    pack_two_bfloat16_into_uint32(std::make_pair(bf_val, bf_val));
            }
        }
    }

    std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(num_tiles * single_tile_size, b_value);

    tt_metal::detail::WriteToBuffer(*src0_dram_buffer, src0_vec);
    tt_metal::detail::WriteToBuffer(*src1_dram_buffer, src1_vec);
    tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(*dst_dram_buffer, result_vec);

    auto u16_src0_vec = u16_from_u32_vector(src0_vec);
    auto u16_src1_vec = u16_from_u32_vector(src1_vec);

    // Compute golden reference for REDUCE_SCALAR across ALL tiles
    // Each tile: b_value × (256×0 + 256×1 + 256×2 + 256×3) = b_value × 1536.0
    // Total: num_tiles × b_value × 1536.0
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

    EXPECT_NEAR(device_scalar, golden_scalar, 0.1f);
}

}  // namespace

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalar) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 1, 1.0f);  // B = 1.0
}

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalarB2) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 1, 2.0f);  // B = 2.0
}

TEST_F(MeshDeviceSingleCardFixture, MulReduceScalarB3) {
    IDevice* device = devices_[0]->get_devices()[0];
    run_mul_reduce_scalar_test(device, 1, 3.0f);  // B = 3.0
}

// TODO: Implement multiple tiles support
// TEST_F(MeshDeviceSingleCardFixture, MulReduceScalar2TilesB3) {
//     IDevice* device = devices_[0]->get_devices()[0];
//     run_mul_reduce_scalar_test(device, 2, 1.0f);  // B = 1.0
// }

}  // namespace mul_reduce_scalar
