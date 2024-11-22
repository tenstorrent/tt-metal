// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <math.h>

#include "device_fixture.hpp"
#include "test_golden_impls.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

// Limits of SFPU input value range. These values are chosen because they
// cover the domains of all SFPU functions used in the test
#define GOLDEN_BOT_LIMIT    (-7.0f)
#define GOLDEN_TOP_LIMIT    (7.0f)
// Small values around zero for domains which have to exclude zeroes, like
// log or reciprocal
#define GOLDEN_NEG_EPSILON  (-0.0001f)
#define GOLDEN_POS_EPSILON  (0.0001f)
// Min/max values of the randomly-generated input block height/width
#define MIN_BLOCK_DIM       (1)
#define MAX_BLOCK_DIM       (10)
// Number of dimensions randomly generated, can be expanded up to 4,
// making an input a full tensor rather than a matrix
#define NUM_DIMS            (2)

using std::map;
using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace tt::tt_metal;

namespace unit_tests::sfpu_util {

// Internal sfpu_op is mapped to proper SFPU function calls
const map<string, map<string, string>> sfpu_op_to_op_name = {
    {"relu", {{"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}},
    {"exponential", {{"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
    {"reciprocal", {{"SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);"}}},
    {"gelu", {{"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"}}},
    {"sqrt", {{"SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);"}}},
    {"sigmoid", {{"SFPU_OP_CHAIN_0", "sigmoid_tile_init(); sigmoid_tile(0);"}}},
    {"log", {{"SFPU_OP_CHAIN_0", "log_tile_init(); log_tile(0);"}}},
    {"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}},
};

// Function that generates different input ranges depending on the SFPU op specified
vector<uint32_t> generate_random_sfpu_vector(
    const float lower,
    const float upper,
    const size_t num_bytes,
    const string& op_name,
    const tt::DataFormat data_format,
    const int seed) {
    if ((op_name == "sqrt") || (op_name == "log")) {
        // sqrt and log have values between (0, upper]
        return unit_tests::compute::generate_random_vector_generalized(GOLDEN_POS_EPSILON, upper, num_bytes, data_format, seed);
    } else if (op_name == "reciprocal") {
        // For reciprocal, exclude zeroes and use (lower, upper) range
        return unit_tests::compute::generate_random_vector_generalized(lower, upper, num_bytes, data_format, seed, true, GOLDEN_NEG_EPSILON, GOLDEN_POS_EPSILON);
    } else {
        // For all other operations, allow zeroes and use (lower, upper) range
        return unit_tests::compute::generate_random_vector_generalized(lower, upper, num_bytes, data_format, seed);
    }
}

// Function that performs SFPU ops on float values.
// It is used to generate golden
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
        return input ? logf(input) : 0.0f;
    } else if (op_name == "tanh") {
        return std::tanh(input);
    } else {
        TT_THROW("Unsupported op_name!");
        return 0.0f;
    }
}

// Function that compares SFPU output and golden. Different tollerances are needed for different ops
bool is_close_packed_sfpu_output(const vector<float>& vec_a, const vector<float>& vec_b, const string& op_name) {
    for (int i = 0; i < vec_a.size(); i++) {
        if (op_name == "tanh") {
            return is_close<float>(vec_a[i], vec_b[i], 0.175f, 0.1f);
        } else if ((op_name == "sqrt") or (op_name == "reciprocal") or (op_name == "exponential")) {
            return is_close<float>(vec_a[i], vec_b[i], 0.06f, 0.002);
        } else {
            return is_close<float>(vec_a[i], vec_b[i], 0.01f, 0.05f);
        }
    }
    return false;
}

}  // namespace unit_tests::sfpu_util

namespace unit_tests::compute::sfpu {

struct SfpuConfig {
    size_t r_tile_dim = 0;
    size_t c_tile_dim = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores = CoreRangeSet();
    std::string sfpu_op = "";
    bool approx_mode = true;
    bool fp32_dest_acc_en = true;
};

/// @brief Does Dram --> Reader --> CB --> Sfpu Compute --> CB --> Writer --> Dram. So far, enqueue APIs only added to
/// grayskull
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_sfpu_test(tt_metal::Device* device, const SfpuConfig& test_config) {
    size_t num_tiles = test_config.r_tile_dim * test_config.c_tile_dim;
    const size_t input_byte_size = num_tiles * tile_size(test_config.l1_input_data_format);
    const size_t output_byte_size = num_tiles * tile_size(test_config.l1_output_data_format);

    tt_metal::Program program = tt_metal::CreateProgram();

    // Create input/output buffers
    tt::tt_metal::InterleavedBufferConfig input_dram_config{
                    .device = device,
                    .size = input_byte_size,
                    .page_size = input_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    tt::tt_metal::InterleavedBufferConfig output_dram_config{
                    .device = device,
                    .size = output_byte_size,
                    .page_size = output_byte_size,
                    .buffer_type = tt::tt_metal::BufferType::DRAM
    };

    auto input_dram_buffer = CreateBuffer(input_dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_buffer = CreateBuffer(output_dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    vector<uint32_t> compute_kernel_args = {
        uint32_t(test_config.c_tile_dim),  // per_core_block_cnt
        uint32_t(test_config.r_tile_dim)   // per_core_block_dim
    };

    // Create packed input
    vector<uint32_t> packed_input = sfpu_util::generate_random_sfpu_vector(
        GOLDEN_BOT_LIMIT,
        GOLDEN_TOP_LIMIT,
        input_byte_size,
        test_config.sfpu_op,
        test_config.l1_input_data_format,
        std::chrono::system_clock::now().time_since_epoch().count()
    );

    // Unpack input to prepare for golden
    vector<float> unpacked_input = unit_tests::compute::unpack_generalized(test_config.l1_input_data_format, packed_input);

    // Golden output, a float vector
    vector<float> golden(unpacked_input.size());
    std::transform(unpacked_input.begin(), unpacked_input.end(), golden.begin(), [&](const float& val) {
        return sfpu_util::sfpu_function(test_config.sfpu_op, val);
    });

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
        tt_metal::CircularBufferConfig l1_input_cb_config = tt_metal::CircularBufferConfig(input_byte_size, {{0, test_config.l1_input_data_format}})
            .set_page_size(0, tile_size(test_config.l1_input_data_format));
        auto l1_input_cb = tt_metal::CreateCircularBuffer(program, core_range, l1_input_cb_config);

        tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(output_byte_size, {{16, test_config.l1_output_data_format}})
            .set_page_size(16, tile_size(test_config.l1_output_data_format));
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

        map<string, string> sfpu_defines = sfpu_util::sfpu_op_to_op_name.at(test_config.sfpu_op);

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
                .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
                .math_approx_mode = test_config.approx_mode,
                .compile_args = compute_kernel_args,
                .defines = sfpu_defines});

        for (const CoreCoord& core_coord : core_range)
        {
            SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
        }
    }

    vector<uint32_t> packed_output;
    tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
    tt_metal::detail::LaunchProgram(device, program);
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, packed_output);

    // Unpack SFPU output to float vector
    vector<float> unpacked_output = unit_tests::compute::unpack_generalized(test_config.l1_output_data_format, packed_output);

    return sfpu_util::is_close_packed_sfpu_output(golden, unpacked_output, test_config.sfpu_op);
}

}  // namespace unit_tests::compute::sfpu

class SingleCoreSingleDeviceSfpuParameterizedFixture : public DeviceFixture,
                                                       public testing::WithParamInterface<std::tuple<std::tuple<tt::DataFormat, tt::DataFormat>, std::string>> {
};

TEST_P(SingleCoreSingleDeviceSfpuParameterizedFixture, SfpuCompute) {
    // Generate random width and height of the input block
    // Can be easily expanded to all dimensions of tensor
    vector<uint32_t> random_shape = generate_uniform_random_vector<uint32_t>(
        MIN_BLOCK_DIM,
        MAX_BLOCK_DIM,
        NUM_DIMS,
        std::chrono::system_clock::now().time_since_epoch().count() // Seed
    );
    size_t r_tile_dim = random_shape[0];
    size_t c_tile_dim = random_shape[1];

    // Extract the tuple of input/output formats and the sfpu_op
    auto formats = std::get<0>(GetParam());
    string sfpu_op = std::get<1>(GetParam());

    // Extract input and output formats from the tuple
    tt::DataFormat input_format = std::get<0>(formats);
    tt::DataFormat output_format = std::get<1>(formats);

    CoreRange core_range({0, 0}, {0, 0});
    CoreRangeSet core_range_set({core_range});
    for (bool approx_mode: {true, false}) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS
            if ((fp32_dest_acc_en == true) && (this->arch_ == tt::ARCH::GRAYSKULL)) continue;
            unit_tests::compute::sfpu::SfpuConfig test_config = {
                .r_tile_dim = r_tile_dim,
                .c_tile_dim = c_tile_dim,
                .l1_input_data_format = input_format,
                .l1_output_data_format = output_format,
                .cores = core_range_set,
                .sfpu_op = sfpu_op,
                .approx_mode = approx_mode,
                .fp32_dest_acc_en = fp32_dest_acc_en
            };
            log_info("SFPU_OP={}, r_tile_dim={}, c_tile_dim={}, approx_mode={}, fp32_dest_acc_en={} input_format={} output_format={}",
                sfpu_op,
                r_tile_dim,
                c_tile_dim,
                approx_mode,
                fp32_dest_acc_en,
                input_format,
                output_format);
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_sfpu_test(devices_.at(id), test_config));
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    SingleCoreSfpuCompute,
    SingleCoreSingleDeviceSfpuParameterizedFixture,
    ::testing::Combine(
        ::testing::Values(
            std::make_tuple(tt::DataFormat::Float16_b, tt::DataFormat::Float16_b),
            std::make_tuple(tt::DataFormat::Float16_b, tt::DataFormat::Float32),
            std::make_tuple(tt::DataFormat::Float32, tt::DataFormat::Float16_b),
            std::make_tuple(tt::DataFormat::Bfp4_b, tt::DataFormat::Float16_b),
            std::make_tuple(tt::DataFormat::Bfp8_b, tt::DataFormat::Float32)
        ),
        ::testing::Values(
            "gelu",
            "relu",
            "sqrt",
            "exponential",
            "log",
            "reciprocal",
            "tanh",
            "sigmoid"
        )
    ));
