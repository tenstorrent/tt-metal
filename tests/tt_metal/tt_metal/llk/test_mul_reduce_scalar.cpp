// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <map>
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
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-logger/tt-logger.hpp>
#include <umd/device/types/arch.hpp>

#include "device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/impl/data_format/bfloat16_utils.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar {

struct MulReduceScalarConfig {
    uint32_t num_tiles = 1;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t seed = 12345;
};

constexpr uint32_t TILE_BYTE_SIZE = 2 * 32 * 32;  // bfloat16: 2 bytes * 32 * 32 elements

bool run_mul_reduce_scalar_test(IDevice* device, const MulReduceScalarConfig& config) {
    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    uint32_t input_buffer_size = config.num_tiles * TILE_BYTE_SIZE;
    tt_metal::InterleavedBufferConfig dram_config = {
        .device = device,
        .size = input_buffer_size,
        .page_size = TILE_BYTE_SIZE,
        .buffer_type = tt_metal::BufferType::DRAM};
    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);

    dram_config.size = TILE_BYTE_SIZE;
    auto dst_dram_buffer = CreateBuffer(dram_config);

    uint32_t cb_tiles = std::max(8u, config.num_tiles);
    uint32_t cb_size = cb_tiles * TILE_BYTE_SIZE;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(cb_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, TILE_BYTE_SIZE);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(cb_size, {{tt::CBIndex::c_1, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_1, TILE_BYTE_SIZE);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    tt_metal::CircularBufferConfig cb_out_config =
        tt_metal::CircularBufferConfig(cb_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, TILE_BYTE_SIZE);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Set up compile-time arguments for the reader kernel using TensorAccessor
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto dual_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary_multiple_tiles.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    const std::map<std::string, std::string> compute_defines = {{"REDUCE_OP", "PoolType::SUM"}};
    auto mul_reduce_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/mul_reduce_scalar.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = config.math_fidelity, .compile_args = {}, .defines = compute_defines});

    SetRuntimeArgs(
        program,
        dual_reader_kernel,
        core,
        {src0_dram_buffer->address(), 0, src1_dram_buffer->address(), 0, config.num_tiles});

    SetRuntimeArgs(program, unary_writer_kernel, core, {dst_dram_buffer->address(), 0, 1});

    SetRuntimeArgs(program, mul_reduce_kernel, core, {config.num_tiles});

    uint32_t byte_size = config.num_tiles * TILE_BYTE_SIZE;
    auto packed_input0 = test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0, 1.0f, byte_size / sizeof(bfloat16), config.seed);

    bfloat16 one_bf16 = bfloat16(1.0f);
    uint16_t one_u16 = std::bit_cast<uint16_t>(one_bf16);
    std::vector<uint32_t> packed_input1(byte_size / sizeof(uint32_t));
    for (uint32_t& val : packed_input1) {
        val = (static_cast<uint32_t>(one_u16) << 16) | one_u16;
    }

    tt_metal::detail::WriteToBuffer(*src0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(*src1_dram_buffer, packed_input1);
    tt_metal::detail::LaunchProgram(device, program, true, true);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(*dst_dram_buffer, result_vec);

    auto u16_src0_vec = u16_from_u32_vector(packed_input0);
    auto u16_src1_vec = u16_from_u32_vector(packed_input1);

    float golden_scalar = 0.0f;
    for (size_t i = 0; i < u16_src0_vec.size(); ++i) {
        float val0 = static_cast<float>(std::bit_cast<bfloat16>(u16_src0_vec[i]));
        float val1 = static_cast<float>(std::bit_cast<bfloat16>(u16_src1_vec[i]));
        golden_scalar += val0 * val1;
    }

    auto u16_result_vec = u16_from_u32_vector(result_vec);
    float device_scalar = static_cast<float>(std::bit_cast<bfloat16>(u16_result_vec[0]));

    log_info(
        LogTest,
        "num_tiles={}: Golden={}, Device={}, Diff={}",
        config.num_tiles,
        golden_scalar,
        device_scalar,
        std::abs(device_scalar - golden_scalar));

    float rel_tol = 0.01f;
    float abs_tol = 0.01f;
    float tolerance = std::max(rel_tol * std::abs(golden_scalar), abs_tol);
    bool pass = std::abs(device_scalar - golden_scalar) < tolerance;

    return pass;
}

}  // namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar

using namespace tt::tt_metal::unit_tests::compute::mul_reduce_scalar;

// Test fixture that automatically skips if not on Blackhole
class MulReduceScalarTest : public MeshDeviceSingleCardFixture, public testing::WithParamInterface<int> {
protected:
    void SetUp() override {
        MeshDeviceSingleCardFixture::SetUp();
        if (this->arch_ != tt::ARCH::BLACKHOLE) {
            GTEST_SKIP() << "Test only runs on Blackhole architecture";
        }
    }
};

// Single parametrized test
TEST_P(MulReduceScalarTest, MulReduceScalar) {
    IDevice* device = devices_[0]->get_devices()[0];
    int num_tiles = GetParam();
    ASSERT_TRUE(run_mul_reduce_scalar_test(device, {.num_tiles = num_tiles}));
}

// Instantiate the test suite with different tile counts
INSTANTIATE_TEST_SUITE_P(
    MulReduceScalarTests,
    MulReduceScalarTest,
    testing::Values(1, 2, 3, 7, 8),
    [](const testing::TestParamInfo<int>& info) { return "MulReduceScalar_" + std::to_string(info.param) + "_Tiles"; });
