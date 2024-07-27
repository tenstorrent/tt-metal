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

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;
using namespace tt::tt_metal;

namespace unit_tests::compute::tilize {

enum UntilizeType : uint8_t {
    UNPACK = 0,
    PACK = 1
};

struct TilizeConfig {
    uint32_t single_tile_size;
    uint32_t num_tiles_r;
    uint32_t num_tiles_c;
    std::optional<UntilizeType> untilize_type = std::nullopt;
    std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const std::vector<uint32_t>&)> golden_function;
};

void run_single_core_tilize_program(tt_metal::Device* device, const TilizeConfig& test_config) {
    Program program = tt::tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t num_tiles = test_config.num_tiles_r * test_config.num_tiles_c;

    uint32_t dram_buffer_size = test_config.single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig dram_config{
            .device=device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt_metal::Buffer> src_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src_addr = src_dram_buffer->address();

    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    CoreCoord dram_src_noc_xy = src_dram_buffer->noc_coordinates();
    CoreCoord dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 8;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * test_config.single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, test_config.single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 8;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * test_config.single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, test_config.single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto unary_reader_kernel = tt_metal::CreateKernel(
        program,
        test_config.untilize_type.has_value() ? "tt_metal/kernels/dataflow/reader_unary.cpp" : "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        1, // per_core_block_cnt
        uint(test_config.num_tiles_c) // per_core_block_tile_cnt
    };

    string compute_kernel;
    if (test_config.untilize_type.has_value()) {
        string untilize_type = magic_enum::enum_name(test_config.untilize_type.value()).data();
        std::transform(untilize_type.begin(), untilize_type.end(), untilize_type.begin(), [](unsigned char c){ return std::tolower(c); });
        compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/" + untilize_type + "_untilize.cpp";
    } else {
        compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/tilize.cpp";
    }

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel,
        core,
        {dram_buffer_src_addr,
        (std::uint32_t)dram_src_noc_xy.x,
        (std::uint32_t)dram_src_noc_xy.y,
        num_tiles});

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {dram_buffer_dst_addr,
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        num_tiles});

    std::vector<uint32_t> src_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);

    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    vector<uint32_t> golden = test_config.golden_function(src_vec, {test_config.num_tiles_r * 32, test_config.num_tiles_c * 32});

    EXPECT_EQ(golden.size(), result_vec.size());
    EXPECT_EQ(golden, result_vec);
}

} // namespace unit_tests::compute::tilize

TEST_F(DeviceFixture, ComputeUnpackTilize) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .golden_function = unit_tests::compute::gold_standard_tilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackUntilize) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilize) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}
