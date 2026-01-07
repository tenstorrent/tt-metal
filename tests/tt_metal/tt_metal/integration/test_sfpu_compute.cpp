// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include "impl/dispatch/command_queue.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include "command_queue_fixture.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <umd/device/types/arch.hpp>
#include <tt-metalium/distributed.hpp>

namespace tt::tt_metal {

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::sfpu_util {

const map<std::string, std::map<std::string, std::string>> sfpu_op_to_op_name = {
    // FIXME: #1157
    {"relu", {{"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}},
    {"exponential", {{"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
    {"reciprocal", {{"SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);"}}},
    {"gelu", {{"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"}}},
    {"sqrt", {{"SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);"}}},
    {"sigmoid", {{"SFPU_OP_CHAIN_0", "sigmoid_tile_init(); sigmoid_tile(0);"}}},
    {"log", {{"SFPU_OP_CHAIN_0", "log_tile_init(); log_tile(0);"}}},
    {"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}},
};

bfloat16 sfpu_function(const std::string& op_name, const bfloat16& input) {
    if (op_name == "relu") {
        return bfloat16(fmaxf(static_cast<float>(input), 0.0f));
    }
    if (op_name == "exponential") {
        return bfloat16(std::exp(static_cast<float>(input)));
    }
    if (op_name == "reciprocal") {
        return bfloat16(1 / static_cast<float>(input));
    }
    if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x = static_cast<float>(input);
        auto x3 = x * x * x;
        float result = x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
        return bfloat16(result);
    }
    if (op_name == "sqrt") {
        return bfloat16(sqrtf(static_cast<float>(input)));
    }
    if (op_name == "sigmoid") {
        auto x = static_cast<float>(input);
        float result = 1 / (1 + std::exp(-x));
        return bfloat16(result);
    }
    if (op_name == "log") {
        return bfloat16(logf(static_cast<float>(input)));
    }
    if (op_name == "tanh") {
        return bfloat16(std::tanh(static_cast<float>(input)));
    }
    TT_THROW("Unsupported op_name in test");
}
vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const std::string& op_name, const int seed) {
    if ((op_name == "sqrt") or (op_name == "log")) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.0001f, 4.0f, numel, seed);
    }
    if ((op_name == "exponential") or (op_name == "gelu") or (op_name == "reciprocal")) {
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    }
    return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed);
}

bool is_close_packed_sfpu_output(
    const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const std::string& op_name) {
    if (op_name == "tanh") {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    }
    if ((op_name == "gelu") or (op_name == "relu")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.15f); });
    }
    if ((op_name == "exponential")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.1f, 0.1f); });
    }
    return is_close_packed_vectors<bfloat16, uint32_t>(
        vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.06f, 0.006f); });
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores;
    std::string sfpu_op;
    bool approx_mode = true;
};

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(distributed::MeshCommandQueue& cq, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::CreateProgram();
    auto mesh_workload = distributed::MeshWorkload();

    const distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = byte_size,
        .buffer_type = tt_metal::BufferType::DRAM,
    };

    const distributed::ReplicatedBufferConfig replicated_buffer_config{.size = byte_size};
    auto input_dram_buffer =
        distributed::MeshBuffer::create(replicated_buffer_config, device_local_config, cq.device());
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto output_dram_buffer =
        distributed::MeshBuffer::create(replicated_buffer_config, device_local_config, cq.device());
    uint32_t output_dram_byte_address = output_dram_buffer->address();

    vector<uint32_t> compute_kernel_args = {
        uint32_t(test_config.num_tiles),  // per_core_block_cnt
        1                                 // per_core_block_cnt
    };

    // Input
    std::vector<uint32_t> packed_input = sfpu_util::generate_packed_sfpu_input(
        byte_size / sizeof(bfloat16), test_config.sfpu_op, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    auto input = unpack_vector<bfloat16, uint32_t>(packed_input);
    std::vector<bfloat16> golden(input.size());
    std::transform(input.begin(), input.end(), golden.begin(), [&](const bfloat16& val) {
        return sfpu_util::sfpu_function(test_config.sfpu_op, val);
    });
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

    // Same runtime args for every core
    vector<uint32_t> reader_rt_args = {
        (uint32_t)input_dram_byte_address,
        0,
        (uint32_t)test_config.num_tiles,
    };

    vector<uint32_t> writer_rt_args = {
        (uint32_t)output_dram_byte_address,
        0,
        (uint32_t)test_config.num_tiles,
    };

    for (const CoreRange& core_range : test_config.cores.ranges()) {
        tt_metal::CircularBufferConfig l1_input_cb_config =
            tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_0, test_config.l1_input_data_format}})
                .set_page_size(tt::CBIndex::c_0, test_config.tile_byte_size);
        tt_metal::CreateCircularBuffer(program, core_range, l1_input_cb_config);

        tt_metal::CircularBufferConfig l1_output_cb_config =
            tt_metal::CircularBufferConfig(byte_size, {{tt::CBIndex::c_16, test_config.l1_output_data_format}})
                .set_page_size(tt::CBIndex::c_16, test_config.tile_byte_size);
        tt_metal::CreateCircularBuffer(program, core_range, l1_output_cb_config);

        auto reader_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary.cpp",
            test_config.cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        // Enqueue apis only supported on gs so far
        auto writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            test_config.cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        std::map<std::string, std::string> sfpu_defines = sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op);

        sfpu_defines["SFPU_OP_EXP_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_GELU_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_RECIP_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_SQRT_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_ELU_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_NEG_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"] = "1";

        tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            test_config.cores,
            tt_metal::ComputeConfig{
                .math_approx_mode = test_config.approx_mode,
                .compile_args = compute_kernel_args,
                .defines = sfpu_defines});

        // TODO(agrebenisan): Clean this up to only use the first path once Enqueue apis supported on WH
        for (const CoreCoord& core_coord : core_range) {
            SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
        }
    }

    mesh_workload.add_program(distributed::MeshCoordinateRange(cq.device()->shape()), std::move(program));

    std::vector<uint32_t> dest_buffer_data;
    distributed::EnqueueWriteMeshBuffer(cq, input_dram_buffer, packed_input, false);

    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);

    distributed::ReadShard(cq, dest_buffer_data, output_dram_buffer, distributed::MeshCoordinate(0, 0));

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu
class SingleCoreSingleCardSfpuParameterizedFixture
    : public UnitMeshCQFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};
TEST_P(SingleCoreSingleCardSfpuParameterizedFixture, TensixSfpuCompute) {
    for (const auto& device_ : devices_) {
        size_t num_tiles = std::get<0>(GetParam());
        std::string sfpu_op = std::get<1>(GetParam());

        if ((arch_ == tt::ARCH::WORMHOLE_B0 or arch_ == tt::ARCH::BLACKHOLE) and sfpu_op == "log") {
            GTEST_SKIP() << "log has very high abs and relative diff";
        }

        CoreRange core_range({0, 0}, {0, 0});
        CoreRangeSet core_range_set({core_range});
        unit_tests::compute::sfpu::SfpuConfig test_config = {
            .num_tiles = num_tiles,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .cores = core_range_set,
            .sfpu_op = sfpu_op,
            .approx_mode = false};
        log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
        EXPECT_TRUE(run_sfpu_all_same_buffer(device_->mesh_command_queue(), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleCardSfpuParameterizedFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh")));
class SingleCoreSingleCardSfpuParameterizedApproxFixture
    : public UnitMeshCQFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(SingleCoreSingleCardSfpuParameterizedApproxFixture, TensixSfpuCompute) {
    for (const auto& device_ : devices_) {
        size_t num_tiles = std::get<0>(GetParam());
        std::string sfpu_op = std::get<1>(GetParam());

        if ((arch_ == tt::ARCH::WORMHOLE_B0 or arch_ == tt::ARCH::BLACKHOLE) and sfpu_op == "log") {
            GTEST_SKIP() << "log has very high abs and relative diff";
        }

        CoreRange core_range({0, 0}, {0, 0});
        CoreRangeSet core_range_set({core_range});
        unit_tests::compute::sfpu::SfpuConfig test_config = {
            .num_tiles = num_tiles,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .cores = core_range_set,
            .sfpu_op = sfpu_op,
            .approx_mode = true};
        log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
        EXPECT_TRUE(run_sfpu_all_same_buffer(device_->mesh_command_queue(), test_config));
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleCardSfpuParameterizedApproxFixture,
    ::testing::Values(
        std::make_tuple(1, "relu"),
        std::make_tuple(1, "exponential"),
        std::make_tuple(1, "reciprocal"),
        std::make_tuple(1, "gelu"),
        std::make_tuple(1, "sqrt"),
        std::make_tuple(1, "sigmoid"),
        std::make_tuple(1, "log"),
        std::make_tuple(1, "tanh"),
        std::make_tuple(4, "relu"),
        std::make_tuple(4, "exponential"),
        std::make_tuple(4, "reciprocal"),
        std::make_tuple(4, "gelu"),
        std::make_tuple(4, "sqrt"),
        std::make_tuple(4, "sigmoid"),
        std::make_tuple(4, "log"),
        std::make_tuple(4, "tanh")));

class MultiCoreSingleCardSfpuParameterizedApproxFixture
    : public UnitMeshCQFixture,
      public testing::WithParamInterface<std::tuple<size_t, std::string>> {};

TEST_P(MultiCoreSingleCardSfpuParameterizedApproxFixture, TensixAllCoreMultiTileSfpuApproxCompute) {
    for (const auto& device_ : devices_) {
        size_t num_tiles = std::get<0>(GetParam());
        std::string sfpu_op = std::get<1>(GetParam());

        if ((arch_ == tt::ARCH::WORMHOLE_B0 or arch_ == tt::ARCH::BLACKHOLE) and sfpu_op == "log") {
            GTEST_SKIP() << "log has very high abs and relative diff";
        }

        CoreCoord worker_grid_size = device_->compute_with_storage_grid_size();
        CoreRange cr({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});
        CoreRangeSet core_range_set({cr});

        unit_tests::compute::sfpu::SfpuConfig test_config = {
            .num_tiles = num_tiles,
            .tile_byte_size = 2 * 32 * 32,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .cores = core_range_set,
            .sfpu_op = sfpu_op,
            .approx_mode = true};
        log_info(tt::LogTest, "Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
        EXPECT_TRUE(run_sfpu_all_same_buffer(device_->mesh_command_queue(), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    MultiCoreSfpuCompute,
    MultiCoreSingleCardSfpuParameterizedApproxFixture,
    ::testing::Values(
        std::make_tuple(20, "relu"),
        std::make_tuple(20, "exponential"),
        std::make_tuple(20, "reciprocal"),
        std::make_tuple(20, "gelu"),
        std::make_tuple(20, "sqrt"),
        std::make_tuple(20, "sigmoid"),
        std::make_tuple(20, "log"),
        std::make_tuple(20, "tanh")));

}  // namespace tt::tt_metal
