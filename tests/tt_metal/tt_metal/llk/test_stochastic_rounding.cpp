// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <map>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
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

/// @brief Test stochastic rounding by verifying statistical properties.
/// For a value between two BF16 representable values, stochastic rounding should
/// probabilistically round up or down such that the expected value equals the true value.
/// @param cq - Mesh command queue
/// @param num_tiles - Number of tiles to process
/// @param fractional_position - Position between two BF16 values (0.0 to 1.0)
///                              e.g., 0.25 means 25% of the way from lower to upper BF16 value
/// @param tolerance - Allowed deviation from expected proportion (for statistical test)
/// @return true if the observed rounding distribution is within tolerance of expected
bool run_stochastic_rounding_statistical_test(
    distributed::MeshCommandQueue& cq, size_t num_tiles, float fractional_position, float tolerance = 0.05f) {
    // BF16 has 7 mantissa bits, so epsilon at 1.0 is 2^-7
    constexpr float bf16_epsilon_at_one = 1.0f / 128.0f;  // 0.0078125
    constexpr float base_value = 1.0f;
    // First value after 1.0 representable in bf16
    const float upper_bf16 = base_value + bf16_epsilon_at_one;

    // float32 representing value between base_value and upper_bf16
    const float test_value = base_value + fractional_position * bf16_epsilon_at_one;

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
        1U                                 // per_core_block_dim
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
        {"SFPU_OP_CHAIN_0", "stochastic_round_tile(0);"},
        {"SFPU_OP_ROUND_FAMILY_INCLUDE", "1"},
    };

    // We want to retain float32 precision when copying tile to DST register
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;

    tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
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
        if (val_float == base_value) {
            count_rounded_down++;
        } else if (val_float == upper_bf16) {
            count_rounded_up++;
        } else {
            // Unexpected value - test should fail
            log_error(
                tt::LogTest, "Unexpected output value: {} (expected {} or {})", val_float, base_value, upper_bf16);
            return false;
        }
    }

    float observed_proportion_up = static_cast<float>(count_rounded_up) / static_cast<float>(num_elements);
    float expected_proportion_up = fractional_position;

    log_info(
        tt::LogTest,
        "Stochastic rounding test: input={}, expected_up={:.2f}%, observed_up={:.2f}% (up={}, down={}, total={})",
        test_value,
        expected_proportion_up * 100.0f,
        observed_proportion_up * 100.0f,
        count_rounded_up,
        count_rounded_down,
        num_elements);

    // Check if observed proportion is within tolerance of expected
    bool pass = std::abs(observed_proportion_up - expected_proportion_up) <= tolerance;
    if (!pass) {
        log_error(
            tt::LogTest,
            "Stochastic rounding distribution outside tolerance: expected={:.2f}%, observed={:.2f}%, tolerance={:.2f}%",
            expected_proportion_up * 100.0f,
            observed_proportion_up * 100.0f,
            tolerance * 100.0f);
    }
    return pass;
}

}  // namespace unit_tests::compute::stochastic_rounding

class StochasticRoundingSingleCardFixture : public UnitMeshCQFixture,
                                            public testing::WithParamInterface<std::tuple<size_t, float>> {};

TEST_P(StochasticRoundingSingleCardFixture, TensixStochasticRoundingStatistical) {
    for (const auto& device_ : devices_) {
        size_t num_tiles = std::get<0>(GetParam());
        float fractional_position = std::get<1>(GetParam());

        log_info(
            tt::LogTest,
            "Testing stochastic rounding: num_tiles={}, fractional_position={:.2f}",
            num_tiles,
            fractional_position);

        // Worst case scenario (fractional_position=0.5) 3σ ≈ 272 elements i.e. ±0.829%
        // The tolerance is much higher because of PRNG and hardware stochastic rounding bugs:
        // https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md
        // https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/TensixTile/TensixCoprocessor/SFPSTOCHRND_FloatFloat.md
        float tolerance = 0.05f;

        EXPECT_TRUE(unit_tests::compute::stochastic_rounding::run_stochastic_rounding_statistical_test(
            device_->mesh_command_queue(), num_tiles, fractional_position, tolerance));
    }
}

// Note: Despite testing stochastic rounding, the results are deterministic, because
// the generic eltwise_sfpu compute kernel doesn't call init_prng_seed(),
// leaving it zero-initialized for every run
INSTANTIATE_TEST_SUITE_P(
    StochasticRoundingCompute,
    StochasticRoundingSingleCardFixture,
    ::testing::Values(
        // Test different fractional positions with enough tiles for statistical significance
        // 32 tiles * 32 * 32 = 32768 elements, giving good statistical power
        std::make_tuple(32U, 0.25f),  // 25% should round up
        std::make_tuple(32U, 0.50f),  // 50% should round up
        std::make_tuple(32U, 0.75f),  // 75% should round up
        // Edge cases
        std::make_tuple(32U, 0.10f),  // 10% should round up
        std::make_tuple(32U, 0.90f)   // 90% should round up
        ));

}  // namespace tt::tt_metal
