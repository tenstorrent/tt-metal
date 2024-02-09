// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include <math.h>

#include "tests/tt_metal/test_utils/df/float32.hpp"
#include "tests/tt_metal/tt_metal/unit_tests/common/device_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tests/tt_metal/test_utils/comparison.hpp"

#include "tests/tt_metal/test_utils/print_helpers.hpp"
#include "tests/tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils::df;
using namespace tt::test_utils;


const map<string, std::map<string, string>> sfpu_op_to_op_name = {
    {"sin", {{"SFPU_OP_CHAIN_0", "sin_tile_init(); sin_tile(0);"}}},
    {"cos", {{"SFPU_OP_CHAIN_0", "cos_tile_init(); cos_tile(0);"}}},
    {"tan", {{"SFPU_OP_CHAIN_0", "tan_tile_init(); tan_tile(0);"}}},
    {"asin", {{"SFPU_OP_CHAIN_0", "asin_tile_init(); asin_tile(0);"}}},
    {"acos", {{"SFPU_OP_CHAIN_0", "acos_tile_init(); acos_tile(0);"}}},
    {"atan", {{"SFPU_OP_CHAIN_0", "atan_tile_init(); atan_tile(0);"}}},
    {"erf", {{"SFPU_OP_CHAIN_0", "erf_tile_init(); erf_tile(0);"}}},
    {"erfc", {{"SFPU_OP_CHAIN_0", "erfc_tile_init(); erfc_tile(0);"}}},
    {"square", {{"SFPU_OP_CHAIN_0", "square_tile_init(); square_tile(0);"}}},
    {"relu", {{"SFPU_OP_CHAIN_0", "relu_tile_init(); relu_tile(0);"}}},
    {"exponential", {{"SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);"}}},
    {"reciprocal", {{"SFPU_OP_CHAIN_0", "recip_tile_init(); recip_tile(0);"}}},
    {"gelu", {{"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"}}},
    {"sqrt", {{"SFPU_OP_CHAIN_0", "sqrt_tile_init(); sqrt_tile(0);"}}},
    {"sigmoid", {{"SFPU_OP_CHAIN_0", "sigmoid_tile_init(); sigmoid_tile(0);"}}},
    {"log", {{"SFPU_OP_CHAIN_0", "log_tile_init(); log_tile(0);"}}},
    {"tanh", {{"SFPU_OP_CHAIN_0", "tanh_tile_init(); tanh_tile(0);"}}},
};

bfloat16 sfpu_function(const string& op_name, const bfloat16& input) {
    if (op_name == "relu") {
        return bfloat16(fmaxf(input.to_float(), 0.0f));
    } else if (op_name == "exponential") {
        return bfloat16(std::exp(input.to_float()));
    } else if (op_name == "reciprocal") {
        return bfloat16(1 / input.to_float());
    } else if (op_name == "gelu") {
        static constexpr float alpha = M_2_SQRTPI * M_SQRT1_2;
        auto x = input.to_float();
        auto x3 = x * x * x;
        float result = x * 0.5 * (1.0 + tanhf(alpha * (x + 0.044715 * x3)));
        return bfloat16(result);
    } else if (op_name == "sqrt") {
        return bfloat16(sqrtf(input.to_float()));
    } else if (op_name == "sigmoid") {
        auto x = input.to_float();
        float result = 1 / (1 + std::exp(-x));
        return bfloat16(result);
    } else if (op_name == "log") {
        return bfloat16(logf(input.to_float()));
    } else if (op_name == "tanh") {
        return bfloat16(std::tanh(input.to_float()));
    } else if (op_name == "sin") {
        return bfloat16(std::sin(input.to_float()));
    } else if (op_name == "cos") {
        return bfloat16(std::cos(input.to_float()));
    } else if (op_name == "asin") {
        return bfloat16(std::asin(input.to_float()));
    } else if (op_name == "acos") {
        return bfloat16(std::acos(input.to_float()));
    } else if (op_name == "atan") {
        return bfloat16(std::atan(input.to_float()));
    } else if (op_name == "erf") {
        return bfloat16(std::erf(input.to_float()));
    } else if (op_name == "erfc") {
        return bfloat16(std::erfc(input.to_float()));
    } else if (op_name == "square") {
        auto x = input.to_float();
        auto x2 = x * x;
        return bfloat16(x2);
    } else if (op_name == "tan") {
        return bfloat16(std::tan(input.to_float()));
    } else {
        TT_THROW("Unsupported op_name in test");
        return bfloat16(0.0f);
    }
}

vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const string& op_name, const int seed) {
    if ((op_name == "sqrt") or (op_name == "log")) {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(0.0001f, 4.0f, numel, seed);
    } else if ((op_name == "exponential") or (op_name == "gelu") or (op_name == "reciprocal")) {
        auto possible_values = vector<bfloat16>({-1.0f, -0.5f, 0.5f, 1.0f});
        return generate_packed_random_vector_from_vector<uint32_t, bfloat16>(possible_values, numel, seed);
    } else {
        return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed);
    }
}

bool is_close_packed_sfpu_output(const vector<uint32_t>& vec_a, const vector<uint32_t>& vec_b, const string& op_name) {
    if (op_name == "tanh") {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.175f, 0.1f); });
    } else if ((op_name == "gelu") or (op_name == "relu")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.15f); });
    } else if ((op_name == "exponential")) {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.1f, 0.1f); });
    } else {
        return is_close_packed_vectors<bfloat16, uint32_t>(
            vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.9f, 0.9f); });
    }
}

struct SfpuConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreRangeSet cores = {{}};
    std::string sfpu_op = "";
    bool approx_mode = true;
};

bool run_sfpu_all_same_buffer(tt_metal::Device* device, const SfpuConfig& test_config) {
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::InterleavedBufferConfig dram_config{
                .device=device,
                .size = byte_size,
                .page_size = byte_size,
                .buffer_type = tt_metal::BufferType::DRAM
                };

    auto input_dram_buffer = CreateBuffer(dram_config);
    uint32_t input_dram_byte_address = input_dram_buffer->address();
    auto input_dram_noc_xy = input_dram_buffer->noc_coordinates();
    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer->address();
    auto output_dram_noc_xy = output_dram_buffer->noc_coordinates();

    vector<uint32_t> compute_kernel_args = {
        uint32_t(test_config.num_tiles),
        1
    };

    std::vector<uint32_t> packed_input = generate_packed_sfpu_input(
        byte_size / bfloat16::SIZEOF, test_config.sfpu_op, std::chrono::system_clock::now().time_since_epoch().count());

    auto input = unpack_vector<bfloat16, uint32_t>(packed_input);
    std::vector<bfloat16> golden(input.size());
    std::transform(input.begin(), input.end(), golden.begin(), [&](const bfloat16& val) {
        return sfpu_function(test_config.sfpu_op, val);
    });
    std::vector<uint32_t> packed_golden = pack_vector<uint32_t, bfloat16>(golden);

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

        std::map<string, string> sfpu_defines = sfpu_op_to_op_name.at(test_config.sfpu_op);

        sfpu_defines["SFPU_OP_EXP_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_GELU_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_RECIP_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_SQRT_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_ERF_ERFC_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_ELU_INCLUDE"] = "1";
        sfpu_defines["SFPU_OP_RELU_FAMILY_INCLUDE"] = "1";
        sfpu_defines["UNARY_MICROKERNEL_PROFILER"] = "1";
        sfpu_defines["SFPU_OP_COMPUTE_KERNEL_API_INCLUDE"]="1";
        sfpu_defines["SFPU_OP_TRIG_FAMILY_INCLUDE"]="1";

        auto sfpu_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            test_config.cores,
            tt_metal::ComputeConfig{
                .math_approx_mode = test_config.approx_mode,
                .compile_args = compute_kernel_args,
                .defines = sfpu_defines});

        int chip_id = 0;
        CoresInCoreRangeGenerator cores_in_core_range(core_range, device->logical_grid_size());

        bool terminate;

        do {
            auto [core_coord, terminate_] = cores_in_core_range();

            terminate = terminate_;

            SetRuntimeArgs(program, writer_kernel, core_coord, writer_rt_args);
            SetRuntimeArgs(program, reader_kernel, core_coord, reader_rt_args);
        } while (not terminate);
    }

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::WriteToBuffer(input_dram_buffer, packed_input);
    tt_metal::detail::LaunchProgram(device, program);
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    return is_close_packed_sfpu_output(dest_buffer_data, packed_golden, test_config.sfpu_op);
}

int main(int argc, char** argv) {
    const int numMainIterations = 1000;

    for (int iteration = 0; iteration < numMainIterations; ++iteration) {
        std::vector<std::tuple<size_t, std::string>> testCases = {
            std::make_tuple(1, "sin"),
            std::make_tuple(1, "cos"),
            std::make_tuple(1, "tan"),
            std::make_tuple(1, "asin"),
            std::make_tuple(1, "square"),
            std::make_tuple(1, "acos"),
            std::make_tuple(1, "atan"),
            std::make_tuple(1, "erf"),
            std::make_tuple(1, "erfc"),
            std::make_tuple(1, "relu"),
            std::make_tuple(1, "exponential"),
            std::make_tuple(1, "reciprocal"),
            std::make_tuple(1, "gelu"),
            std::make_tuple(1, "sqrt"),
            std::make_tuple(1, "sigmoid"),
            std::make_tuple(1, "log"),
            std::make_tuple(1, "tanh"),
        };

        for (const auto& testCase : testCases) {
            size_t num_tiles = std::get<0>(testCase);
            std::string sfpu_op = std::get<1>(testCase);
            CoreRange core_range({0, 0}, {0, 0});
            CoreRangeSet core_range_set({core_range});
            SfpuConfig test_config = {
                .num_tiles = num_tiles,
                .tile_byte_size = 1 * 32 * 32,
                .l1_input_data_format = tt::DataFormat::Float16_b,
                .l1_output_data_format = tt::DataFormat::Float16_b,
                .cores = core_range_set,
                .sfpu_op = sfpu_op,
                .approx_mode = false
            };
            log_info("Testing SFPU_OP={} num_tiles={}", sfpu_op, num_tiles);

                for (unsigned int id = 0; id < 1; id++) {
                    tt::tt_metal::Device* device = tt::tt_metal::CreateDevice(id);
                    bool result = run_sfpu_all_same_buffer(device, test_config);
                    if (result) {
                        std::cout << "Test passed on device " << id << std::endl;
                    } else {
                        std::cout << "Test failed on device " << id << std::endl;
                    }
                    tt::tt_metal::CloseDevice(device);
                }
        }
    }

    return 0;
}
