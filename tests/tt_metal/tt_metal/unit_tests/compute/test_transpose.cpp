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
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "test_golden_impls.hpp"
#include "common/test_tiles.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::compute::transpose {

enum TransposeType : uint8_t {
    WH = 0
};

struct TransposeConfig {
    bool short_init;
    uint32_t single_tile_size;
    std::vector<uint32_t> shape;
    TransposeType transpose_type;
};

void validate_transpose_wh(const std::vector<uint32_t> &src_vec, const std::vector<uint32_t> &shape, const std::vector<uint32_t> &result_vec) {
    int argfail = -1;
    auto comparison_function = [](float a, float b) {
        const float rtol = 0.02f;
        const float atol = 1e-3f;
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
        if (!result)
            absdiff *= 1.0f; // breakpoint spot
        return result;
    };

    // recover a linear view of input vector for consumption by gold_ function
    auto u16_src0_vec = u16_from_u32_vector(src_vec);
    vector<uint16_t> src_linear = convert_layout<uint16_t>(u16_src0_vec, shape, TensorLayout::TILED32_4FACES, TensorLayout::LIN_ROW_MAJOR);
    vector<uint16_t> gold_reduced = gold_transpose_wh(src_linear, shape); // result is uint16_t untilized

    // Tilize from row major and convert to pairs (uint32_t)
    TT_FATAL(shape.size() == 4);
    vector<uint32_t> shapeR{shape[0], shape[1], shape[3], shape[2]};
    auto gold_4f_u32 = u32_from_u16_vector(convert_layout<uint16_t>(gold_reduced, shapeR, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED32_4FACES));

    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    if (not pass) {
        log_error(LogTest, "Failure position={}", argfail);
    }
    EXPECT_TRUE(pass);
}

void run_single_core_transpose(tt_metal::Device* device, const TransposeConfig& test_config) {
    TT_FATAL(test_config.shape.size() == 4);

    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t W = test_config.shape[3], H = test_config.shape[2], NC = test_config.shape[1]*test_config.shape[0];
    uint32_t HW = H*W;
    TT_FATAL(W % 32 == 0 && H % 32 == 0);
    TT_FATAL(H > 0 && W > 0 && NC > 0);
    uint32_t Wt = W/32;
    // size of DST register, with unary r/w this currently only works if the entire Wt fits into DST for reduce
    TT_FATAL(Wt <= 16);
    uint32_t Ht = H/32;
    float scaler = 1.0f/W;
    uint32_t num_tensor_tiles = NC*H*W / (32*32);

    uint32_t dram_buffer_size = test_config.single_tile_size * num_tensor_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
            .device=device,
            .size = dram_buffer_size,
            .page_size = test_config.single_tile_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt_metal::Buffer> src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    CoreCoord dram_src_noc_xy = src_dram_buffer->noc_coordinates();
    CoreCoord dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_buffer_tiles = 32;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_buffer_tiles * test_config.single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, test_config.single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_buffer_tiles = 32;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_buffer_tiles * test_config.single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, test_config.single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(Ht*Wt*NC)
    };

    std::map<string, string> defines = {};

    if (test_config.short_init)
    {
        defines["SHORT_INIT"] = "1";
    }

    auto transpose_compute_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/transpose_wh.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = defines}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel,
        core,
        {
            dram_buffer_src_addr,
            (std::uint32_t)dram_src_noc_xy.x,
            (std::uint32_t)dram_src_noc_xy.y,
            num_tensor_tiles, NC, Ht, Wt, Ht*Wt,
            0 /* no scaler */
        }
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {
            dram_buffer_dst_addr,
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            num_tensor_tiles
        }
    );

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    vector<uint32_t> src_vec = create_random_vector_of_bfloat16(dram_buffer_size, 100.0f, 0x1234);
    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(result_vec.size(), NC*H*W/2); // we are expecting one tile in H, and half the elements since the vector packs 2 uint16_ts

    validate_transpose_wh(src_vec, test_config.shape, result_vec);
}

} // namespace unit_tests::compute::transpose

TEST_F(DeviceFixture, ComputeTransposeWH) {
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = false,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3*32*1, 4*32*1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeTransposeWHShortInit) {
    unit_tests::compute::transpose::TransposeConfig test_config = {
        .short_init = true,
        .single_tile_size = 2 * 1024,
        .shape = {1, 3, 3*32*1, 4*32*1},
        .transpose_type = unit_tests::compute::transpose::TransposeType::WH};
    unit_tests::compute::transpose::run_single_core_transpose(this->devices_.at(0), test_config);
}
