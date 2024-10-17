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

#define GOLDEN_BOT_LIMIT (-7.0f)
#define GOLDEN_TOP_LIMIT (7.0f)
#define GOLDEN_NEG_EPSILON (-0.0001f)
#define GOLDEN_POS_EPISLON (0.0001f)

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

template <typename T>
T sfpu_function(const string& op_name, const T& input);

template <>
tt::test_utils::df::bfloat16 sfpu_function(const string& op_name, const tt::test_utils::df::bfloat16& input) {
    float value = input.to_float();

    if (op_name == "relu") {
        return tt::test_utils::df::bfloat16(fmaxf(value, 0.0f));
    } else if (op_name == "exponential") {
        return tt::test_utils::df::bfloat16(std::exp(value));
    } else if (op_name == "reciprocal") {
        return tt::test_utils::df::bfloat16(1 / value);
    } else if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x3 = value * value * value;
        float result = value * 0.5 * (1.0 + tanhf(alpha * (value + 0.044715 * x3)));
        return tt::test_utils::df::bfloat16(result);
    } else if (op_name == "sqrt") {
        return tt::test_utils::df::bfloat16(sqrtf(value));
    } else if (op_name == "sigmoid") {
        float result = 1 / (1 + std::exp(-value));
        return tt::test_utils::df::bfloat16(result);
    } else if (op_name == "log") {
        return tt::test_utils::df::bfloat16(logf(value));
    } else if (op_name == "tanh") {
        return tt::test_utils::df::bfloat16(std::tanh(value));
    } else {
        TT_THROW("Unsupported op_name in test");
        return tt::test_utils::df::bfloat16(0.0f);
    }
}

template <>
float sfpu_function(const string& op_name, const float& input) {
    if (op_name == "relu") {
        return fmaxf(input, 0.0f);
    } else if (op_name == "exponential") {
        return std::exp(input);
    } else if (op_name == "reciprocal") {
        return 1 / input;
    } else if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x3 = input * input * input;
        return input * 0.5 * (1.0 + tanhf(alpha * (input + 0.044715 * x3)));
    } else if (op_name == "sqrt") {
        return sqrtf(input);
    } else if (op_name == "sigmoid") {
        return 1 / (1 + std::exp(-input));
    } else if (op_name == "log") {
        return logf(input);
    } else if (op_name == "tanh") {
        return std::tanh(input);
    } else {
        TT_THROW("Unsupported op_name in test");
        return 0.0f;
    }
}


// Helper function to generate a vector without zeroes
vector<uint32_t> generate_non_zero_vector(const float lower, const float upper, const unsigned int numel, const int seed) {
    vector<uint32_t> vec;
    // Split into negative and positive parts, avoiding zero
    auto negative_part = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(lower, GOLDEN_NEG_EPSILON, numel / 2, seed);
    auto positive_part = generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(GOLDEN_POS_EPISLON, upper, numel - (numel / 2), seed + 1);

    // Combine both parts
    vec.insert(vec.end(), negative_part.begin(), negative_part.end());
    vec.insert(vec.end(), positive_part.begin(), positive_part.end());
    return vec;
}

vector<uint32_t> generate_packed_sfpu_input(const float lower, const float upper, const unsigned int numel, const string& op_name, const int seed) {
    if ((op_name == "sqrt") || (op_name == "log")) {
        // sqrt and log have values between (0, upper]
        return generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(GOLDEN_POS_EPISLON, upper, numel, seed);
    } else if (op_name == "reciprocal") {
        // For reciprocal, exclude zero and use (upper, lower) range
        return generate_non_zero_vector(lower, upper, numel, seed);
    } else {
        // For all other operations, allow zero and use (upper, lower) range
        return generate_packed_uniform_random_vector<uint32_t, tt::test_utils::df::bfloat16>(lower, upper, numel, seed);
    }
}

bool is_close_packed_sfpu_output(const std::vector<uint32_t>& vec_a, const std::vector<uint32_t>& vec_b, const string& op_name) {
    if (op_name == "tanh") {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    } else if ((op_name == "sqrt") or (op_name == "reciprocal") or (op_name == "exponential")) {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.06f); });
    } else {
        return is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            vec_a, vec_b, [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) { return is_close(a, b, 0.01f, 0.025f); });
    }
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t r_tile_dim = 0;
    size_t c_tile_dim = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores = {{}};
    std::string sfpu_op = "";
    bool approx_mode = true;
    bool fp32_dest_acc_en = true;
};

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_all_same_buffer(tt_metal::Device* device, const SfpuConfig& test_config) {
    size_t num_tiles = test_config.r_tile_dim * test_config.c_tile_dim;
    const size_t byte_size = num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::CreateProgram();
    tt::tt_metal::InterleavedBufferConfig dram_config{
                    .device = device,
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
        uint32_t(test_config.c_tile_dim),  // per_core_block_cnt
        uint32_t(test_config.r_tile_dim)   // per_core_block_dim
    };

    // Input
    std::vector<uint32_t> packed_input = sfpu_util::generate_packed_sfpu_input(
        GOLDEN_BOT_LIMIT,
        GOLDEN_TOP_LIMIT,
        byte_size / tt::datum_size(test_config.l1_input_data_format),
        test_config.sfpu_op,
        std::chrono::system_clock::now().time_since_epoch().count());

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
        (uint32_t)num_tiles,
    };

    vector<uint32_t> writer_rt_args = {
        (uint32_t)output_dram_byte_address,
        (uint32_t)output_dram_noc_xy.x,
        (uint32_t)output_dram_noc_xy.y,
        (uint32_t)num_tiles,
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

        if (test_config.fp32_dest_acc_en) {
            sfpu_defines["DEST_ACCUM_EN"] = "1";
        }

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
    tt_metal::detail::LaunchProgram(device, program);
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    return sfpu_util::is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu

class SingleCoreSingleDeviceSfpuParameterizedFixture : public DeviceFixture,
                                                       public testing::WithParamInterface<std::string> {
};
TEST_P(SingleCoreSingleDeviceSfpuParameterizedFixture, SfpuCompute) {
    vector<uint32_t> random_shape = generate_uniform_random_vector<uint32_t>(
        1,  // Min val
        10, // Max val
        2,  // Number of elements
        std::chrono::system_clock::now().time_since_epoch().count() // Seed
    );
    size_t r_tile_dim = random_shape[0];
    size_t c_tile_dim = random_shape[1];
    string sfpu_op = GetParam();

    // Define supported_formats as a vector or initializer list
    std::vector<tt::DataFormat> supported_formats = {
        tt::DataFormat::Float16_b,
        // tt::DataFormat::Bfp8_b,
        // tt::DataFormat::Float32
    };

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    for (bool approx_mode: {true, false}) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS
            if ((fp32_dest_acc_en == true) && (this->arch_ == tt::ARCH::GRAYSKULL)) continue;
            for (tt::DataFormat format : supported_formats) {
                unit_tests::compute::sfpu::SfpuConfig test_config = {
                    .r_tile_dim = r_tile_dim,
                    .c_tile_dim = c_tile_dim,
                    .tile_byte_size = tile_size(format),
                    .l1_input_data_format = format,
                    .l1_output_data_format = format,
                    .cores = core_range_set,
                    .sfpu_op = sfpu_op,
                    .approx_mode = approx_mode,
                    .fp32_dest_acc_en = fp32_dest_acc_en
                };
                log_info("Testing SFPU_OP={}, r_tile_dim={}, c_tile_dim={}, approx_mode={}, fp32_dest_acc_en={}",
                    sfpu_op,
                    r_tile_dim,
                    c_tile_dim,
                    approx_mode,
                    fp32_dest_acc_en);
                for (unsigned int id = 0; id < num_devices_; id++) {
                    EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(id), test_config));
                }
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleDeviceSfpuParameterizedFixture,
    ::testing::Values(
        "gelu",
        "relu",
        "sqrt",
        "exponential",
        "log",
        "reciprocal",
        "tanh",
        "sigmoid"
        ));
// TEST_F(DeviceFixture, DISABLED_MultiContinguousCoreSingleTileSfpuApproxCompute) {
//     CoreRange core_range({0, 0}, {1, 0});
//     CoreRangeSet core_range_set({core_range});
//     unit_tests::compute::sfpu::SfpuConfig test_config = {
//         .tile_byte_size = 2 * 32 * 32,
//         .l1_input_data_format = tt::DataFormat::Float16_b,
//         .l1_output_data_format = tt::DataFormat::Float16_b,
//         .cores = core_range_set,
//         .approx_mode = true};

//     auto arch = this->arch_;

//     if (arch != tt::ARCH::GRAYSKULL) {
//         GTEST_SKIP();
//     }

//     CoreRangeSet core_set({core_range});
//     test_config.cores = core_set;

//     test_config.r_tile_dim = random_shape[0];
//     test_config.c_tile_dim = random_shape[1];
//     test_config.sfpu_op = "relu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "exponential";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "reciprocal";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "gelu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sqrt";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sigmoid";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "log";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "tanh";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
// }

// TEST_F(DeviceFixture, DISABLED_MultiContinguousCoreMultiTileSfpuApproxCompute) {
//     CoreRange core_range({0, 0}, {1, 0});
//     CoreRangeSet core_range_set({core_range});
//     unit_tests::compute::sfpu::SfpuConfig test_config = {
//         .tile_byte_size = 2 * 32 * 32,
//         .l1_input_data_format = tt::DataFormat::Float16_b,
//         .l1_output_data_format = tt::DataFormat::Float16_b,
//         .cores = core_range_set,
//         .approx_mode = true};

//     auto arch = this->arch_;

//     if (arch != tt::ARCH::GRAYSKULL) {
//         GTEST_SKIP();
//     }

//     CoreRangeSet core_set({core_range});
//     test_config.cores = core_set;

//     test_config.r_tile_dim = random_shape[0];
//     test_config.c_tile_dim = random_shape[1];

//     test_config.sfpu_op = "relu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "exponential";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "reciprocal";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "gelu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sqrt";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sigmoid";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "log";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "tanh";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
// }
// TEST_F(DeviceFixture, DISABLED_AllCoreSingleTileSfpuApproxCompute) {
//     unit_tests::compute::sfpu::SfpuConfig test_config = {
//         .tile_byte_size = 2 * 32 * 32,
//         .l1_input_data_format = tt::DataFormat::Float16_b,
//         .l1_output_data_format = tt::DataFormat::Float16_b,
//         .cores = {{}},
//         .approx_mode = true};

//     auto arch = this->arch_;

//     if (arch != tt::ARCH::GRAYSKULL) {
//         GTEST_SKIP();
//     }

//     int chip_id = 0;
//     CoreCoord worker_grid_size = this->devices_.at(0)->logical_grid_size();
//     CoreRange core_range({0, 0}, {worker_grid_size.x - 2, worker_grid_size.y - 2});

//     CoreRangeSet core_set({core_range});
//     test_config.cores = core_set;

//     test_config.r_tile_dim = random_shape[0];
//     test_config.c_tile_dim = random_shape[1];
//     test_config.sfpu_op = "relu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "exponential";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "reciprocal";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "gelu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sqrt";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sigmoid";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "log";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "tanh";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
// }
// TEST_F(DeviceFixture, DISABLED_AllCoreMultiTileSfpuApproxCompute) {
//     unit_tests::compute::sfpu::SfpuConfig test_config = {
//         .tile_byte_size = 2 * 32 * 32,
//         .l1_input_data_format = tt::DataFormat::Float16_b,
//         .l1_output_data_format = tt::DataFormat::Float16_b,
//         .cores = {{}},
//         .approx_mode = true};

//     auto arch = this->arch_;

//     if (arch != tt::ARCH::GRAYSKULL) {
//         GTEST_SKIP();
//     }

//     int chip_id = 0;
//     CoreCoord worker_grid_size = this->devices_.at(0)->logical_grid_size();
//     CoreRange core_range({0, 0}, {worker_grid_size.x - 2, worker_grid_size.y - 2});

//     CoreRangeSet core_set({core_range});
//     test_config.cores = core_set;
//     test_config.r_tile_dim = random_shape[0];
//     test_config.c_tile_dim = random_shape[1];
//     test_config.sfpu_op = "relu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "exponential";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "reciprocal";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "gelu";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sqrt";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "sigmoid";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "log";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
//     test_config.sfpu_op = "tanh";
//     EXPECT_TRUE(run_sfpu_all_same_buffer(devices_.at(0), test_config));
// }
