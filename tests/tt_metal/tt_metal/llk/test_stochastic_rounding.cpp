// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-logger/tt-logger.hpp>
#include "tt_metal/test_utils/packing.hpp"
#include <umd/device/types/arch.hpp>

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::compute::stochastic_rounding {

struct StochasticRoundingConfig {
    size_t num_tiles = 0;
    uint32_t seed = 0;
    float base_value = 0.0f;
    float fraction = 0.5f;  // Position between base and next BF16 value (0.0 = base, 1.0 = next)
};

struct StochasticRoundingResult {
    size_t count_rounded_up = 0;
    size_t count_rounded_down = 0;
    size_t total = 0;
};

/// @brief Get the next representable BF16 value away from zero (1 ULP)
float bf16_next(float value) {
    bfloat16 bf = bfloat16(value);
    uint16_t bits = std::bit_cast<uint16_t>(bf);
    bits += 1;
    bfloat16 result = std::bit_cast<bfloat16>(bits);
    return static_cast<float>(result);
}

/// @brief Run stochastic rounding and return the results.
/// @param cq - Mesh command queue
/// @param test_config - Test configuration
/// @return StochasticRoundingResult with counts of rounded up/down values
StochasticRoundingResult run_stochastic_rounding(
    distributed::MeshCommandQueue& cq, const StochasticRoundingConfig& test_config) {
    size_t num_tiles = test_config.num_tiles;
    uint32_t seed = test_config.seed;
    float base_value = test_config.base_value;
    float fraction = test_config.fraction;

    bfloat16 base_bf16 = bfloat16(base_value);
    float base_bf16_float = static_cast<float>(base_bf16);
    float upper_bf16 = bf16_next(base_bf16_float);

    // Test value: at 'fraction' position between base and upper BF16
    // This is a float32 value that cannot be exactly represented in BF16
    const float test_value = base_bf16_float + ((upper_bf16 - base_bf16_float) * fraction);

    const size_t tile_byte_size_input = 4U * tt::constants::TILE_HW;   // Float32: 4 bytes per element
    const size_t tile_byte_size_output = 2U * tt::constants::TILE_HW;  // BFloat16: 2 bytes per element
    const size_t input_byte_size = num_tiles * tile_byte_size_input;
    const size_t output_byte_size = num_tiles * tile_byte_size_output;
    const size_t num_elements = num_tiles * tt::constants::TILE_HW;

    tt_metal::Program program = tt_metal::CreateProgram();
    auto mesh_workload = distributed::MeshWorkload();

    const distributed::DeviceLocalBufferConfig input_device_local_config{
        .page_size = input_byte_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };
    const distributed::DeviceLocalBufferConfig output_device_local_config{
        .page_size = output_byte_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };

    const distributed::ReplicatedBufferConfig input_replicated_config{.size = input_byte_size};
    const distributed::ReplicatedBufferConfig output_replicated_config{.size = output_byte_size};

    auto input_dram_buffer =
        distributed::MeshBuffer::create(input_replicated_config, input_device_local_config, cq.device());
    auto output_dram_buffer =
        distributed::MeshBuffer::create(output_replicated_config, output_device_local_config, cq.device());

    uint32_t input_dram_byte_address = input_dram_buffer->address();
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    std::vector<uint32_t> packed_input(num_elements);
    uint32_t test_value_bits = std::bit_cast<uint32_t>(test_value);
    std::fill(packed_input.begin(), packed_input.end(), test_value_bits);

    std::vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(num_tiles),  // per_core_block_cnt
        seed                               // seed for PRNG
    };

    std::vector<uint32_t> reader_rt_args = {
        input_dram_byte_address,
        0,
        static_cast<uint32_t>(num_tiles),
    };

    std::vector<uint32_t> writer_rt_args = {
        output_dram_byte_address,
        0,
        static_cast<uint32_t>(num_tiles),
    };

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet cores({core_range});

    // Input CB: Float32
    tt_metal::CircularBufferConfig l1_input_cb_config =
        tt_metal::CircularBufferConfig(input_byte_size, {{tt::CBIndex::c_0, tt::DataFormat::Float32}})
            .set_page_size(tt::CBIndex::c_0, tile_byte_size_input);
    tt_metal::CreateCircularBuffer(program, core_range, l1_input_cb_config);

    // Output CB: BFloat16
    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(output_byte_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, tile_byte_size_output);
    tt_metal::CreateCircularBuffer(program, core_range, l1_output_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary.cpp",
        cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    std::map<std::string, std::string> sfpu_defines = {
        {"SFPU_OP_ROUND_FAMILY_INCLUDE", "1"},
    };

    // We want to retain float32 precision when copying tile to DST register
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/stochastic_rounding_sfpu.cpp",
        cores,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = true,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = sfpu_defines});

    for (const CoreCoord& core_coord : core_range) {
        SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
        SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
    }

    mesh_workload.add_program(distributed::MeshCoordinateRange(cq.device()->shape()), std::move(program));

    distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, packed_input, false);
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);

    std::vector<uint32_t> dest_buffer_data;
    distributed::ReadShard(cq, dest_buffer_data, output_dram_buffer, distributed::MeshCoordinate(0, 0));

    // Unpack BFloat16 results and count how many rounded up vs down
    auto output = unpack_vector<bfloat16, uint32_t>(dest_buffer_data);

    size_t count_rounded_up = 0;
    size_t count_rounded_down = 0;

    for (const auto& val : output) {
        float val_float = static_cast<float>(val);
        if (val_float == base_bf16_float) {
            count_rounded_down++;
        } else if (val_float == upper_bf16) {
            count_rounded_up++;
        } else {
            log_error(
                tt::LogTest, "Unexpected output value: {} (expected {} or {})", val_float, base_bf16_float, upper_bf16);
        }
    }

    log_info(
        tt::LogTest,
        "Stochastic rounding: input={}, base={}, upper={}, fraction={}, seed={}, up={}, down={}, total={}",
        test_value,
        base_bf16_float,
        upper_bf16,
        fraction,
        seed,
        count_rounded_up,
        count_rounded_down,
        num_elements);

    return StochasticRoundingResult{
        .count_rounded_up = count_rounded_up,
        .count_rounded_down = count_rounded_down,
        .total = num_elements,
    };
}

}  // namespace unit_tests::compute::stochastic_rounding

using namespace unit_tests::compute::stochastic_rounding;

class StochasticRoundingSingleCardFixture : public UnitMeshCQFixture,
                                            public testing::WithParamInterface<StochasticRoundingConfig> {};

TEST_P(StochasticRoundingSingleCardFixture, TensixStochasticRoundingCorrectness) {
    StochasticRoundingConfig test_config = GetParam();

    log_info(
        tt::LogTest,
        "Testing stochastic rounding: num_tiles={}, seed={}, base_value={}",
        test_config.num_tiles,
        test_config.seed,
        test_config.base_value);

    auto result = run_stochastic_rounding(devices_.at(0)->mesh_command_queue(), test_config);
    EXPECT_EQ(result.count_rounded_down + result.count_rounded_up, result.total);
}

INSTANTIATE_TEST_SUITE_P(
    StochasticRoundingCompute,
    StochasticRoundingSingleCardFixture,
    ::testing::Values(
        // Test with different seeds and base values
        // 128 tiles * 32 * 32 = 131072 elements
        StochasticRoundingConfig{
            .num_tiles = 128U, .seed = 111U, .base_value = 0.0f},  // Note: subnormals get flushed to zero
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 222U, .base_value = 1.0f},
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 444U, .base_value = 0.5f},
        StochasticRoundingConfig{
            .num_tiles = 128U, .seed = 55555U, .base_value = 256.0f},  // Next representable value in BF16 is 258.0
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 555U, .base_value = -0.5f}));

class StochasticRoundingDistributionFixture : public UnitMeshCQFixture,
                                              public testing::WithParamInterface<StochasticRoundingConfig> {};

TEST_P(StochasticRoundingDistributionFixture, TensixStochasticRoundingDistribution) {
    StochasticRoundingConfig test_config = GetParam();
    constexpr float tolerance = 0.05f;

    log_info(
        tt::LogTest,
        "Testing stochastic rounding distribution: num_tiles={}, seed={}, base_value={}, fraction={}",
        test_config.num_tiles,
        test_config.seed,
        test_config.base_value,
        test_config.fraction);

    auto result = run_stochastic_rounding(devices_.at(0)->mesh_command_queue(), test_config);

    float actual_ratio = static_cast<float>(result.count_rounded_up) / static_cast<float>(result.total);
    float expected_ratio = test_config.fraction;
    float deviation = std::abs(actual_ratio - expected_ratio);

    log_info(
        tt::LogTest,
        "Distribution check: expected_ratio={}, actual_ratio={}, deviation={}, tolerance={}",
        expected_ratio,
        actual_ratio,
        deviation,
        tolerance);

    EXPECT_LE(deviation, tolerance) << "Stochastic rounding distribution mismatch: expected " << expected_ratio
                                    << " (+/- " << tolerance << "), got " << actual_ratio;
}

INSTANTIATE_TEST_SUITE_P(
    StochasticRoundingDistribution,
    StochasticRoundingDistributionFixture,
    ::testing::Values(
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 12345U, .base_value = 1.0f, .fraction = 0.5f},
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 23456U, .base_value = 1.0f, .fraction = 0.25f},
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 34567U, .base_value = 1.0f, .fraction = 0.75f},
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 45678U, .base_value = 1.0f, .fraction = 0.1f},
        StochasticRoundingConfig{.num_tiles = 128U, .seed = 56789U, .base_value = 1.0f, .fraction = 0.9f}));

}  // namespace tt::tt_metal
