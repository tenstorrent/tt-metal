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
    PACK = 1,
    DST = 2
};

enum TilizeType : uint8_t {
    UNPACK_A = 0,
    UNPACK_A_B = 1,
};

// TilizeA_B takes 2 input source vectors instead of one
using GoldenFunc = std::variant<
    std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const std::vector<uint32_t>&)>,
    std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&)> >;

struct TestConfig {
    bool short_init = false;
    uint32_t single_tile_size;
    uint32_t num_tiles_r;
    uint32_t num_tiles_c;
    std::optional<UntilizeType> untilize_type = std::nullopt;
    std::optional<TilizeType> tilize_type = std::nullopt;
    GoldenFunc golden_function;
};

void run_single_core_tilize_program(tt_metal::Device* device, const TestConfig& test_config) {
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

    std::shared_ptr<tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    CoreCoord dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    CoreCoord dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t num_input_tiles = num_tiles;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * test_config.single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, test_config.single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    std::shared_ptr<tt_metal::Buffer> src1_dram_buffer;
    uint32_t dram_buffer_src1_addr;
    CoreCoord dram_src1_noc_xy;

    if(test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        src1_dram_buffer = CreateBuffer(dram_config);
        dram_buffer_src1_addr = src1_dram_buffer->address();
        dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();

        uint32_t src1_cb_index = tt::CB::c_in1;
        uint32_t num_input_tiles = num_tiles;
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * test_config.single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, test_config.single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
    }

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * test_config.single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, test_config.single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    string reader_kernel_path;
    if(test_config.untilize_type.has_value()){
        reader_kernel_path = "tt_metal/kernels/dataflow/reader_unary.cpp";
    } else if(test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        reader_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp";
    } else {
        reader_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_4.cpp";
    }

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        reader_kernel_path,
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {
        uint(test_config.num_tiles_r), // per_core_block_cnt
        uint(test_config.num_tiles_c) // per_core_block_tile_cnt
    };

    string compute_kernel;
    if (test_config.untilize_type.has_value()) {
        string untilize_type = magic_enum::enum_name(test_config.untilize_type.value()).data();
        std::transform(untilize_type.begin(), untilize_type.end(), untilize_type.begin(), [](unsigned char c){ return std::tolower(c); });
        compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/" + untilize_type + "_untilize.cpp";
    } else if (test_config.tilize_type.has_value()) {
        compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/";
        compute_kernel += (test_config.tilize_type == TilizeType::UNPACK_A) ? "tilize.cpp" : "unpack_tilizeA_B.cpp";
    } else {
        tt::log_fatal("Invalid untilize and tilize type value");
    }

    std::map<string, string> defines = {};

    if (test_config.short_init)
    {
        defines["SHORT_INIT"] = "1";
    }

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = defines}
    );

    std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(dram_buffer_size, false);
    tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

    std::vector<uint32_t> src1_vec;

    if(test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {
                dram_buffer_src0_addr,
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                dram_buffer_src1_addr,
                (std::uint32_t)dram_src1_noc_xy.x,
                (std::uint32_t)dram_src1_noc_xy.y,
                (uint32_t)num_tiles,
            });

        src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 1.0f);
        tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);

    } else {
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tiles
            });
    }

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {dram_buffer_dst_addr,
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        num_tiles});

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    vector<uint32_t> golden;
    vector<uint32_t> shape = {test_config.num_tiles_r * 32, test_config.num_tiles_c * 32};
    bool pass = true;

    //Call golden function with correct number of parameters depending on test
    std::visit([&](auto&& func) {
    using FuncType = std::decay_t<decltype(func)>;
        if constexpr (std::is_same_v<FuncType, std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const std::vector<uint32_t>&)> >) {
            golden = func(src0_vec, shape);
        } else if constexpr (std::is_same_v<FuncType, std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&)> >) {
            golden = func(src0_vec, src1_vec, shape);
        } else {
            log_fatal("Invalid golden function type");
        }
    }, test_config.golden_function);

    if(test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        pass &= (golden.size() == result_vec.size());
        pass &= is_close_packed_vectors<tt::test_utils::df::bfloat16, uint32_t>(
            result_vec,
            golden,
            [&](const tt::test_utils::df::bfloat16& a, const tt::test_utils::df::bfloat16& b) {
                return is_close(a, b, 0.01f);
            });

    } else {
        pass &= (golden.size() == result_vec.size());
        pass &= (golden == result_vec);
    }

    // if (not pass){
    //     std::cout << "GOLDEN "  << std::endl;
    //     print_vector(unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(golden));
    //     std::cout << "RESULTS "  << std::endl;
    //     print_vector(unpack_vector<tt::test_utils::df::bfloat16, uint32_t>(result_vec));
    // }
    ASSERT_TRUE(pass);
}

} // namespace unit_tests::compute::tilize

TEST_F(DeviceFixture, ComputeUnpackTilize1x4) {
    unit_tests::compute::tilize::TestConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
        .golden_function = unit_tests::compute::gold_standard_tilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackTilize2x2) {
    unit_tests::compute::tilize::TestConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 2,
        .num_tiles_c = 2,
        .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
        .golden_function = unit_tests::compute::gold_standard_tilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackTilize4x1) {
    unit_tests::compute::tilize::TestConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 4,
        .num_tiles_c = 1,
        .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
        .golden_function = unit_tests::compute::gold_standard_tilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackTilizeA_B) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    unit_tests::compute::tilize::TestConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 2,
        .num_tiles_c = 8,
        .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A_B,
        .golden_function = unit_tests::compute::gold_standard_tilize_w_elwadd
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}


TEST_F(DeviceFixture, ComputeUnpackUntilize1x4) {
    unit_tests::compute::tilize::TestConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilize1x4) {
    unit_tests::compute::tilize::TestConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackUntilize2x2) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 2,
        .num_tiles_c = 2,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilize2x2) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 2,
        .num_tiles_c = 2,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackUntilize4x1) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 4,
        .num_tiles_c = 1,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilize4x1) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 4,
        .num_tiles_c = 1,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilizeShortInit1x4) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .short_init = true,
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilizeShortInit2x2) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .short_init = true,
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 2,
        .num_tiles_c = 2,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilizeShortInit4x1) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .short_init = true,
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 4,
        .num_tiles_c = 1,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackUntilizeShortInit1x4) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .short_init = true,
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackUntilizeShortInit2x2) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .short_init = true,
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 2,
        .num_tiles_c = 2,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputeUnpackUntilizeShortInit4x1) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .short_init = true,
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 4,
        .num_tiles_c = 1,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilizeDst1x4) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 1,
        .num_tiles_c = 4,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilizeDst2x2) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 2,
        .num_tiles_c = 2,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}

TEST_F(DeviceFixture, ComputePackUntilizeDst4x1) {
    unit_tests::compute::tilize::TilizeConfig test_config = {
        .single_tile_size = 2 * 1024,
        .num_tiles_r = 4,
        .num_tiles_c = 1,
        .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
        .golden_function = unit_tests::compute::gold_standard_untilize
    };
    unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
}
