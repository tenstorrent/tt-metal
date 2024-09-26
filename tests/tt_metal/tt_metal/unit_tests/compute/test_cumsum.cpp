// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "test_golden_impls.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::compute::cumsum {

struct CumsumConfig {
    int N;
    int Wt;
    int Ht;
    bool rowwise;
};

std::vector<tt::test_utils::df::bfloat16> gold_cumsum(std::vector<tt::test_utils::df::bfloat16>& src, const std::vector<uint32_t> &shape, bool rowwise) {
    int N = shape.at(0);
    int W = shape.at(1);
    int H = shape.at(2);

    std::vector<tt::test_utils::df::bfloat16> golden(N * W * H);

    int dim_a = rowwise ? H : W;
    int dim_b = rowwise ? W : H;
    int j_mul = rowwise ? 1 : W;
    int k_mul = rowwise ? W : 1;

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < dim_a; k++) {
            float res = 0;
            for (int j = 0; j < dim_b; j++) {
                res += src[i * W * H + j * j_mul + k * k_mul].to_float();
                golden[i * W * H + j * j_mul + k * k_mul] = res;
            }
        }
    }

    return golden;
}

void run_single_core_cumsum(tt_metal::Device* device, const CumsumConfig& test_config) {
    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;

    constexpr uint32_t single_tile_size = tile_width * tile_height * tt::test_utils::df::bfloat16::SIZEOF;

    uint32_t W = test_config.Wt * tile_width;
    uint32_t H = test_config.Ht * tile_height;
    uint32_t dram_buffer_size = single_tile_size * test_config.N * test_config.Wt * test_config.Ht;

    tt_metal::InterleavedBufferConfig dram_config{
        .device=device,
        .size = dram_buffer_size,
        .page_size = dram_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM
    };

    auto src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();
    auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();
    tt_metal::CircularBufferConfig l1_src_cb_config = tt_metal::CircularBufferConfig(dram_buffer_size, {{0, tt::DataFormat::Float16_b}})
        .set_page_size(0, single_tile_size);
    auto l1_src_cb = tt_metal::CreateCircularBuffer(program, core, l1_src_cb_config);

    auto dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();
    auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();
    tt_metal::CircularBufferConfig l1_dst_cb_config = tt_metal::CircularBufferConfig(dram_buffer_size, {{16, tt::DataFormat::Float16_b}})
        .set_page_size(16, single_tile_size);
    auto l1_dst_cb = tt_metal::CreateCircularBuffer(program, core, l1_dst_cb_config);

    string reader_kernel_name, writer_kernel_name;
    std::map<string, string> defines = {};
    std::vector<uint32_t> compile_args = {};

    if(test_config.rowwise) {
        reader_kernel_name = "tt_metal/kernels/dataflow/reader_unary.cpp";
        writer_kernel_name = "tt_metal/kernels/dataflow/writer_unary.cpp";
        compile_args = {test_config.Wt, test_config.Ht, test_config.N};
        defines["ROWWISE"] = "1";
    } else {
        reader_kernel_name = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh.cpp";
        writer_kernel_name = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_transpose_wh.cpp";
        compile_args = {test_config.Ht, test_config.Wt, test_config.N};
    }

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        reader_kernel_name,
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto writer_kernel = tt_metal::CreateKernel(
        program,
        writer_kernel_name,
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto compute_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/cumsum.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compile_args, .defines = defines});

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {
            (uint32_t)dram_buffer_src_addr,
            (uint32_t)src_dram_noc_xy.x,
            (uint32_t)src_dram_noc_xy.y,
            (uint32_t)test_config.N * test_config.Ht * test_config.Wt, // Used for non transposing kernel
            (uint32_t)test_config.N,                                   // Used for transposing kernel
            (uint32_t)test_config.Ht,                                  // Used for transposing kernel
            (uint32_t)test_config.Wt,                                  // Used for transposing kernel
            (uint32_t)test_config.Ht * test_config.Wt                  // Used for transposing kernel
        });

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        core,
        {
            (uint32_t)dram_buffer_dst_addr,
            (uint32_t)dst_dram_noc_xy.x,
            (uint32_t)dst_dram_noc_xy.y,
            (uint32_t)test_config.N * test_config.Ht * test_config.Wt, // Used for non transposing kernel
            (uint32_t)test_config.N,                                   // Used for transposing kernel
            (uint32_t)test_config.Ht,                                  // Used for transposing kernel
            (uint32_t)test_config.Wt,                                  // Used for transposing kernel
            (uint32_t)test_config.Ht * test_config.Wt                  // Used for transposing kernel
        });

    std::vector<tt::test_utils::df::bfloat16> input = generate_uniform_random_vector<tt::test_utils::df::bfloat16>(
        -1.0f,
        1.0f,
        dram_buffer_size / tt::test_utils::df::bfloat16::SIZEOF,
        std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<tt::test_utils::df::bfloat16> golden = gold_cumsum(input, {test_config.N, W, H}, test_config.rowwise);
    auto golden_packed = pack_vector<uint32_t, tt::test_utils::df::bfloat16>(golden);

    auto input_packed = pack_vector<uint32_t, tt::test_utils::df::bfloat16>(input);
    auto input_packed_tilized = unit_tests::compute::gold_standard_tilize(input_packed, {test_config.N * test_config.Ht, test_config.Wt});

    tt_metal::detail::WriteToBuffer(src_dram_buffer, input_packed_tilized);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> output_packed_tilized;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, output_packed_tilized);
    auto output_packed = unit_tests::compute::gold_standard_untilize(output_packed_tilized, {test_config.N * test_config.Ht, test_config.Wt});

    log_info(tt::LogTest, "Running test for N = {}, Wt = {}, Ht = {}", test_config.N, test_config.Wt, test_config.Ht);

    bool result = is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
        output_packed,
        golden_packed,
        [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) {
            return is_close(a, b, 0.01f);
        });
    ASSERT_TRUE(result);
}
}

TEST_F(DeviceFixture, ComputeCumsumColumnwise) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP(); // Not implemented for GRAYSKULL
    }

    for (int i = 1; i <= 3; i++) {
        for (int j = 1; j <= 3; j++) {
            for (int k = 1; k <= 3; k++) {
                unit_tests::compute::cumsum::CumsumConfig test_config =
                {
                    .N = i,
                    .Wt = j,
                    .Ht = k,
                    .rowwise = false
                };
                unit_tests::compute::cumsum::run_single_core_cumsum(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeCumsumRowwise) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP(); // Not implemented for GRAYSKULL
    }

    for (int i = 1; i <= 3; i++) {
        for (int j = 1; j <= 3; j++) {
            for (int k = 1; k <= 3; k++) {
                unit_tests::compute::cumsum::CumsumConfig test_config =
                {
                    .N = i,
                    .Wt = j,
                    .Ht = k,
                    .rowwise = true
                };
                unit_tests::compute::cumsum::run_single_core_cumsum(this->devices_.at(0), test_config);
            }
        }
    }
}
