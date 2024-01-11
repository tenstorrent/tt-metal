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

#include "test_tiles.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::test_utils::df;
using namespace tt::test_utils;


const map<string, std::map<string, string>> sfpu_op_to_op_name = {
    {"dropout", {{"SFPU_OP_CHAIN_0", "dropout_tile_init(); dropout_tile(0);"}}},
};

vector<uint32_t> generate_packed_sfpu_input(const unsigned int numel, const string& op_name, const int seed) {
    return generate_packed_uniform_random_vector<uint32_t, bfloat16>(-1.0f, 1.0f, numel, seed);
}

bool is_close_packed_sfpu_output(const vector<uint32_t>& vec_a, const vector<uint32_t>& vec_b, const string& op_name) {
    return is_close_packed_vectors<bfloat16, uint32_t>(
        vec_a, vec_b, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.9f, 0.9f); });
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

inline void count_zero(std::vector<bfloat16> &vec, int num_tiles) {
    int idx = 0;
    int zero = 0;
    for (int i = 0; i < num_tiles; i++) {
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                if (vec.at(idx).to_uint16() == 0) {
                    zero++;
                }
                idx++;
            }
        }
    }
    log_info(LogTest, "{} zero elements in a tile. ", zero);
    log_info(LogTest, "droput rate : {}%", static_cast<float>(zero) / 1024);
}

inline void print_vec_of_bfloat16_with_align(std::vector<bfloat16> vec, int num_tiles, string name = "", int tile_print_offset = 0) {
    int idx = 0;
    for (int i = 0; i < num_tiles; i++) {
        std::cout << name << " tile " << i + tile_print_offset << std::endl;
        for (int j = 0; j < 32; j++) {
            for (int k = 0; k < 32; k++) {
                std::cout << std::setw(3) << vec.at(idx).to_float() << ", " ;
                idx++;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

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
    uint32_t input_dram_byte_address = input_dram_buffer.address();
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = CreateBuffer(dram_config);
    uint32_t output_dram_byte_address = output_dram_buffer.address();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    vector<uint32_t> compute_kernel_args = {
        uint32_t(test_config.num_tiles),
        1
    };

    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tile_layout = convert_to_tile_layout(tensor.get_values());
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);

    print_vec_of_bfloat16(tensor.get_values(), 1, "input");

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
            .set_page_size(0, test_config.tile_byte_size );
        auto l1_input_cb = tt_metal::CreateCircularBuffer(program, core_range, l1_input_cb_config);

        tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(byte_size, {{16, test_config.l1_output_data_format}})
            .set_page_size(16, test_config.tile_byte_size );
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
        sfpu_defines["SFPU_OP_DROPOUT_INCLUDE"] = "1";
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
    tt_metal::detail::WriteToBuffer(input_dram_buffer, activations);
    tt_metal::detail::LaunchProgram(device, program);
    tt_metal::DumpDeviceProfileResults(device, program);
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);

    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(dest_buffer_data);
    auto result_flat_layout = convert_to_flat_layout(result_bfp16);

    print_vec_of_bfloat16_with_align(result_flat_layout, 1, "output");
    count_zero(result_flat_layout, 1);

    return true;
}

int main(int argc, char** argv) {
    const int numMainIterations = 1;

    for (int iteration = 0; iteration < numMainIterations; ++iteration) {
        std::vector<std::tuple<size_t, std::string>> testCases = {
            std::make_tuple(1, "dropout"),
        };
        for (const auto& testCase : testCases) {
            size_t num_tiles = std::get<0>(testCase);
            std::string sfpu_op = std::get<1>(testCase);
            CoreRange core_range = {.start = {0, 0}, .end = {0, 0}};
            CoreRangeSet core_range_set({core_range});
            SfpuConfig test_config = {
                .num_tiles = num_tiles,
                .tile_byte_size = 32 * 32 * 2,
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
                    tt::tt_metal::CloseDevice(device);
                }
        }
    }
    return 0;
}
