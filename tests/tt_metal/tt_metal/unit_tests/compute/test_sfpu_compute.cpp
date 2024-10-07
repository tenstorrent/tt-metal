// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <math.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/impl/program/program_pool.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::sfpu_util {


const map<string, std::map<string, string>> sfpu_op_to_op_name = {
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

tt::test_utils::df::bfloat16 sfpu_function(const string& op_name, const tt::test_utils::df::bfloat16& input) {
    if (op_name == "relu") {
        return tt::test_utils::df::bfloat16(fmaxf(input.to_float(), 0.0f));
    } else if (op_name == "exponential") {
        return tt::test_utils::df::bfloat16(std::exp(input.to_float()));
    } else if (op_name == "reciprocal") {
        return tt::test_utils::df::bfloat16(1 / input.to_float());
    } else if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x = input.to_float();
        auto x3 = x * x * x;
        float result = x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
        return tt::test_utils::df::bfloat16(result);
    } else if (op_name == "sqrt") {
        return tt::test_utils::df::bfloat16(sqrtf(input.to_float()));
    } else if (op_name == "sigmoid") {
        auto x = input.to_float();
        float result = 1 / (1 + std::exp(-x));
        return tt::test_utils::df::bfloat16(result);
    } else if (op_name == "log") {
        return tt::test_utils::df::bfloat16(logf(input.to_float()));
    } else if (op_name == "tanh") {
        return tt::test_utils::df::bfloat16(std::tanh(input.to_float()));
    } else {
        TT_THROW("Unsupported op_name in test");
        return tt::test_utils::df::bfloat16(0.0f);
    }
}
vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const string& op_name, const int seed) {
    if ((op_name == "sqrt") or (op_name == "log")) {
        return generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(0.0001f, 4.0f, numel, seed);
    } else if ((op_name == "exponential") or (op_name == "gelu") or (op_name == "reciprocal")) {
        auto possible_values = vector<tt::test_utils::df::bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, tt::test_utils::df::bfloat16>(possible_values, numel, seed);
    } else {
        return generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(-1.0f, 1.0f, numel, seed);
    }
}

bool is_close_packed_sfpu_output(const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const string& op_name) {
    if (op_name == "tanh") {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    } else if ((op_name == "gelu") or (op_name == "relu")) {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.15f); });
    } else if ((op_name == "exponential")) {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.1f, 0.1f); });
    } else if ((op_name == "log")) {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.03f, 0.02f); });
    } else {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.06f, 0.006f); });
    }
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores = {{}};
    std::string sfpu_op = "";
    bool approx_mode = true;
};

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(tt_metal::Device* device, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    auto program = tt_metal::CreateScopedProgram();
    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device=device,
                    .size = byte_size,
                    .page_size = byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
        };

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    vector<uint32_t> compute_kernel_args = {
        uint32_t(test_config.num_tiles),  // per_core_block_cnt
        1                            // per_core_block_cnt
    };

    // Input
    std::vector<uint32_t> packed_input = sfpu_util::generate_packed_sfpu_input(
        byte_size / tt::test_utils::df::bfloat16::SIZEOF, test_config.sfpu_op, std::chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    auto input = unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(packed_input);
    std::vector<tt::test_utils::df::bfloat16> golden(input.size());
    std::transform(input.begin(), input.end(), golden.begin(), [&](const tt::test_utils::df::bfloat16& val) {
        return sfpu_util::sfpu_function(test_config.sfpu_op, val);
    });
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, tt::test_utils::df::bfloat16>(golden);

    // Same runtime args for every core
    vector<uint32_t> reader_rt_args = {
        (uint32_t)input_dram_byte_address,
        (uint32_t)input_dram_noc_xy.x,
        (uint32_t)input_dram_noc_xy.y,
        (uint32_t)test_config.num_tiles,
    };

    vector<uint32_t> writer_rt_args = {
        (uint32_t)output_dram_byte_address,
        (uint32_t)output_dram_noc_xy.x,
        (uint32_t)output_dram_noc_xy.y,
        (uint32_t)test_config.num_tiles,
    };

    for (const CoreRange& core_range : test_config.cores.ranges()) {
        tt_metal::CircularBufferConfig l1_input_cb_config = tt_metal::CircularBufferConfig(byte_size, {{0, test_config.l1_input_data_format}})
            .set_page_size(0, test_config.tile_byte_size);
        auto l1_input_cb = tt_metal::CreateCircularBuffer(program, core_range, l1_input_cb_config);

        tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(byte_size, {{16, test_config.l1_output_data_format}})
            .set_page_size(16, test_config.tile_byte_size);
        auto l1_output_cb = tt_metal::CreateCircularBuffer(program, core_range, l1_output_cb_config);

        auto reader_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary.cpp",
            test_config.cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            test_config.cores,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        std::map<string, string> sfpu_defines = sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op);

        sfpu_defines["SFPU_OP_EXP_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_GELU_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_RECIP_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_SQRT_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_ELU_INCLUDE"] = "1";
	    sfpu_defines["SFPU_OP_NEG_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"]="1";

        auto sfpu_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            test_config.cores,
            tt_metal::ComputeConfig{
                .math_approx_mode = test_config.approx_mode,
                .compile_args = compute_kernel_args,
                .defines = sfpu_defines});

        for (const CoreCoord& core_coord : core_range)
        {
            SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
        }
    }

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
    auto* program_ptr = tt::tt_metal::ProgramPool::instance().get_program(program);
    tt_metal::detail::LaunchProgram(device, *program_ptr);
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu
class SingleCoreSingleDeviceSfpuParameterizedFixture : public DeviceFixture,
                                                       public testing::WithParamInterface<std::tuple<size_t, string>> {
};
TEST_P(SingleCoreSingleDeviceSfpuParameterizedFixture, SfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    string sfpu_op = std::get<1>(GetParam());

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
    log_info("Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleDeviceSfpuParameterizedFixture,
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
class SingleCoreSingleDeviceSfpuParameterizedApproxFixture
    : public DeviceFixture,
      public testing::WithParamInterface<std::tuple<size_t, string>> {};

TEST_P(SingleCoreSingleDeviceSfpuParameterizedApproxFixture, SfpuCompute) {
    size_t num_tiles = std::get<0>(GetParam());
    string sfpu_op = std::get<1>(GetParam());

    if (((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "relu")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "exponential")) or
        ((arch_ == tt::ARCH::WORMHOLE_B0) and (sfpu_op == "log"))) {
        GTEST_SKIP();
    } else {
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
        log_info("Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);
        for (unsigned int id = 0; id < num_devices_; id++) {
            EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
        }
    }
}
INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleDeviceSfpuParameterizedApproxFixture,
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

TEST_F(DeviceFixture, DISABLED_MultiContinguousCoreSingleTileSfpuApproxCompute) {
    CoreRange core_range({0, 0}, {1, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .approx_mode = true};

    auto arch = this->arch_;

    if (arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    CoreRangeSet core_set({core_range});
    test_config.cores = core_set;

    test_config.num_tiles = 1;
    test_config.sfpu_op = "relu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "exponential";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "reciprocal";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "gelu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sqrt";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sigmoid";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "log";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "tanh";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
}

TEST_F(DeviceFixture, DISABLED_MultiContinguousCoreMultiTileSfpuApproxCompute) {
    CoreRange core_range({0, 0}, {1, 0});
    CoreRangeSet core_range_set({core_range});
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = core_range_set,
        .approx_mode = true};

    auto arch = this->arch_;

    if (arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    CoreRangeSet core_set({core_range});
    test_config.cores = core_set;

    test_config.num_tiles = 4;

    test_config.sfpu_op = "relu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "exponential";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "reciprocal";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "gelu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sqrt";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sigmoid";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "log";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "tanh";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
}
TEST_F(DeviceFixture, DISABLED_AllCoreSingleTileSfpuApproxCompute) {
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = {{}},
        .approx_mode = true};

    auto arch = this->arch_;

    if (arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    int chip_id = 0;
    CoreCoord worker_grid_size = this->devices_.at(0)->logical_grid_size();
    CoreRange core_range({0, 0}, {worker_grid_size.x - 2, worker_grid_size.y - 2});

    CoreRangeSet core_set({core_range});
    test_config.cores = core_set;

    test_config.num_tiles = 1;
    test_config.sfpu_op = "relu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "exponential";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "reciprocal";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "gelu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sqrt";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sigmoid";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "log";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "tanh";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
}
TEST_F(DeviceFixture, DISABLED_AllCoreMultiTileSfpuApproxCompute) {
    unit_tests::compute::sfpu::SfpuConfig test_config = {
        .tile_byte_size = 2 * 32 * 32,
        .l1_input_data_format = tt::DataFormat::Float16_b,
        .l1_output_data_format = tt::DataFormat::Float16_b,
        .cores = {{}},
        .approx_mode = true};

    auto arch = this->arch_;

    if (arch != tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }

    int chip_id = 0;
    CoreCoord worker_grid_size = this->devices_.at(0)->logical_grid_size();
    CoreRange core_range({0, 0}, {worker_grid_size.x - 2, worker_grid_size.y - 2});

    CoreRangeSet core_set({core_range});
    test_config.cores = core_set;
    test_config.num_tiles = 4;
    test_config.sfpu_op = "relu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "exponential";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "reciprocal";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "gelu";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sqrt";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "sigmoid";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "log";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
    test_config.sfpu_op = "tanh";
    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
}
