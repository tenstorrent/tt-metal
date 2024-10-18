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
    std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const GoldenConfig &config)>,
    std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const std::vector<uint32_t>&, const GoldenConfig &config)> >;

struct TestConfig {
    // Whether or not to use *_init_short LLK API calls:
    bool short_init = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    // Whether or not we want the result to be stored in DST in FP32 is
    // controlled with this flag:
    bool fp32_dest_acc_en = false;
    uint32_t input_single_tile_size;
    uint32_t output_single_tile_size;
    // Block height in tiles:
    uint32_t num_tiles_r;
    // Block width in tiles:
    uint32_t num_tiles_c;
    uint32_t num_faces_per_tile = 4;
    // Face height in datums:
    uint32_t face_r_dim = 16;
    std::optional<UntilizeType> untilize_type = std::nullopt;
    std::optional<TilizeType> tilize_type = std::nullopt;
    GoldenFunc golden_function;
};

void run_single_core_tilize_program(tt_metal::Device* device, const TestConfig& test_config) {
    Program program = tt::tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t num_tiles = test_config.num_tiles_r * test_config.num_tiles_c;
    uint32_t input_dram_buffer_size = test_config.input_single_tile_size * num_tiles;
    uint32_t output_dram_buffer_size = test_config.output_single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig input_dram_config{
            .device=device,
            .size = input_dram_buffer_size,
            .page_size = input_dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    tt_metal::InterleavedBufferConfig output_dram_config{
            .device=device,
            .size = output_dram_buffer_size,
            .page_size = output_dram_buffer_size,
            .buffer_type = tt_metal::BufferType::DRAM
    };

    std::shared_ptr<tt_metal::Buffer> src0_dram_buffer = CreateBuffer(input_dram_config);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(output_dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    CoreCoord dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    CoreCoord dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = tt::CB::c_in0;
    uint32_t num_input_tiles = num_tiles;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * test_config.input_single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, test_config.input_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    std::shared_ptr<tt_metal::Buffer> src1_dram_buffer;
    uint32_t dram_buffer_src1_addr;
    CoreCoord dram_src1_noc_xy;

    if(test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        src1_dram_buffer = CreateBuffer(input_dram_config);
        dram_buffer_src1_addr = src1_dram_buffer->address();
        dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();

        uint32_t src1_cb_index = tt::CB::c_in1;
        uint32_t num_input_tiles = num_tiles;
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * test_config.input_single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, test_config.input_single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);
    }

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(
        num_output_tiles * test_config.output_single_tile_size,
        {{ouput_cb_index, test_config.fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b}})
        .set_page_size(ouput_cb_index, test_config.output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    string reader_kernel_path;
    if(test_config.untilize_type.has_value()){
        reader_kernel_path = "tt_metal/kernels/dataflow/reader_unary.cpp";
    } else if(test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        reader_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_binary.cpp";
    } else {
        reader_kernel_path = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp";
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
        if (test_config.untilize_type == UntilizeType::DST) {
            compute_kernel_args.push_back(test_config.num_faces_per_tile);
            compute_kernel_args.push_back(test_config.face_r_dim);
        }
    } else if (test_config.tilize_type.has_value()) {
        compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/";
        compute_kernel += (test_config.tilize_type == TilizeType::UNPACK_A) ? "tilize.cpp" : "unpack_tilizeA_B.cpp";
    } else {
        tt::log_fatal("Invalid untilize and tilize type value");
    }

    std::map<string, string> defines = {};

    if (test_config.short_init) {
        defines["SHORT_INIT"] = "1";
    }
    if (test_config.fp32_dest_acc_en) {
        defines["DST_ACCUM_MODE"] = "1";
    }

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = defines}
    );

    std::vector<uint32_t> src0_vec = create_arange_vector_of_bfloat16(input_dram_buffer_size, false);
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

        src1_vec = create_constant_vector_of_bfloat16(input_dram_buffer_size, 1.0f);
        tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);

    } else {
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel,
            core,
            {dram_buffer_src0_addr,
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            num_tiles,
            src0_cb_index,
            test_config.num_tiles_c,
            false
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
    GoldenConfig config = {
        .num_tiles_r_dim = test_config.num_tiles_r,
        .num_tiles_c_dim = test_config.num_tiles_c,
        .face_r_dim = test_config.face_r_dim,
        .face_c_dim = 16,
        .num_faces = test_config.num_faces_per_tile,
    };
    bool pass = true;

    //Call golden function with correct number of parameters depending on test
    std::visit([&](auto&& func) {
    using FuncType = std::decay_t<decltype(func)>;
        if constexpr (std::is_same_v<FuncType, std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const GoldenConfig &config)> >) {
            golden = func(src0_vec, config);
        } else if constexpr (std::is_same_v<FuncType, std::function<std::vector<uint32_t>(const std::vector<uint32_t>&, const std::vector<uint32_t>&, const GoldenConfig &config)> >) {
            golden = func(src0_vec, src1_vec, config);
        } else {
            log_fatal("Invalid golden function type");
        }
    }, test_config.golden_function);


    if(test_config.fp32_dest_acc_en) {
        vector<bfloat16> golden_unpacked = unpack_vector<bfloat16, uint32_t>(golden);
        // Increasing the size since from BFP16 two times, since storing is in FP32
        golden.resize(golden.size() * 2);
        for (auto i = 0; i < golden_unpacked.size(); i++) {
            // Cast float32 to "packed "uint32 golden vector if fp32_dest_acc_en:
            golden[i] = std::bit_cast<uint32_t>(golden_unpacked[i].to_float());
        }
    }

    if(test_config.tilize_type.has_value() && test_config.tilize_type == TilizeType::UNPACK_A_B) {
        pass &= (golden.size() == result_vec.size());
        pass &= is_close_packed_vectors<bfloat16, uint32_t>(
            result_vec,
            golden,
            [&](const bfloat16& a, const bfloat16& b) {
                return is_close(a, b, 0.01f);
            });

    } else {
        pass &= (golden.size() == result_vec.size());
        pass &= (golden == result_vec);
    }

    if (not pass){
        std::cout << "GOLDEN "  << std::endl;
        print_vector(unpack_vector<bfloat16, uint32_t>(golden));
        std::cout << "RESULTS "  << std::endl;
        print_vector(unpack_vector<bfloat16, uint32_t>(result_vec));
    }
    ASSERT_TRUE(pass);
    log_info(tt::LogTest, "Done running test with: num_tiles_r = {}, num_tiles_c = {}, FP32_DestAcc = {}, DstSyncFull = {}, pass = {}",
            test_config.num_tiles_r,
            test_config.num_tiles_c,
            test_config.fp32_dest_acc_en,
            test_config.dst_full_sync_en,
            pass);
}

} // namespace unit_tests::compute::tilize

/**************************************
Following tests are for Unpack Tilize
***************************************/

TEST_F(DeviceFixture, ComputeUnpackTilize) {
    vector<vector<uint32_t> > num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for(auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS and unpack_tilize hangs on BH -> tt-metal/#13640
            if ((fp32_dest_acc_en == true) && (this->arch_ != tt::ARCH::WORMHOLE_B0)) continue;
            for (bool dst_full_sync_en : {true, false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                    .golden_function = unit_tests::compute::gold_standard_tilize
                };
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeUnpackTilizeA_B) {
    auto arch = this->arch_;
    if (arch == tt::ARCH::GRAYSKULL) {
        GTEST_SKIP();
    }
    for (bool dst_full_sync_en : {true, false}) {
        unit_tests::compute::tilize::TestConfig test_config = {
            .dst_full_sync_en = dst_full_sync_en,
            .input_single_tile_size = 2 * 1024,
            .output_single_tile_size = 2 * 1024,
            .num_tiles_r = 2,
            .num_tiles_c = 8,
            .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A_B,
            .golden_function = unit_tests::compute::gold_standard_tilize_w_elwadd
        };
        unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, ComputeUnpackTilizeShortInit) {
    vector<vector<uint32_t> > num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for(auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS and unpack_tilize hangs on BH -> tt-metal/#13640
            if ((fp32_dest_acc_en == true) && (this->arch_ != tt::ARCH::WORMHOLE_B0)) continue;
            for (bool dst_full_sync_en : {true, false}) {
            unit_tests::compute::tilize::TestConfig test_config = {
                .short_init = true,
                .dst_full_sync_en = dst_full_sync_en,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .input_single_tile_size = 2 * 1024,
                .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .tilize_type = unit_tests::compute::tilize::TilizeType::UNPACK_A,
                .golden_function = unit_tests::compute::gold_standard_tilize
            };
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

/**************************************
Following tests are for Unpack Untilize
***************************************/

TEST_F(DeviceFixture, ComputeUnpackUntilize) {
    vector<vector<uint32_t> > num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for(auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS
            if ((fp32_dest_acc_en == true) && (this->arch_ == tt::ARCH::GRAYSKULL)) continue;
            for (bool dst_full_sync_en : {true, false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
                    .golden_function = unit_tests::compute::gold_standard_untilize
                };
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputeUnpackUntilizeShortInit) {
    vector<vector<uint32_t> > num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for(auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS
            if ((fp32_dest_acc_en == true) && (this->arch_ == tt::ARCH::GRAYSKULL)) continue;
            for (bool dst_full_sync_en : {true, false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .short_init = true,
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .untilize_type = unit_tests::compute::tilize::UntilizeType::UNPACK,
                    .golden_function = unit_tests::compute::gold_standard_untilize
                };
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

/**************************************
Following tests are for pack untilize
***************************************/
TEST_F(DeviceFixture, ComputePackUntilize) {
    vector<vector<uint32_t> > num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for(auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS
            if ((fp32_dest_acc_en == true) && (this->arch_ == tt::ARCH::GRAYSKULL)) continue;
            for (bool dst_full_sync_en : {true, false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
                    .golden_function = unit_tests::compute::gold_standard_untilize
                };
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputePackUntilizeShortInit) {
    vector<vector<uint32_t> > num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for(auto num_tile : num_tiles) {
        for (bool fp32_dest_acc_en : {true, false}) {
            // FP32 dest acc not possible for GS
            if ((fp32_dest_acc_en == true) && (this->arch_ == tt::ARCH::GRAYSKULL)) continue;
            for (bool dst_full_sync_en : {true, false}) {
                unit_tests::compute::tilize::TestConfig test_config = {
                    .short_init = true,
                    .dst_full_sync_en = dst_full_sync_en,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .input_single_tile_size = 2 * 1024,
                    .output_single_tile_size = 1024 * (fp32_dest_acc_en ? 4 : 2),
                    .num_tiles_r = num_tile[0],
                    .num_tiles_c = num_tile[1],
                    .untilize_type = unit_tests::compute::tilize::UntilizeType::PACK,
                    .golden_function = unit_tests::compute::gold_standard_untilize
                };
                unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
            }
        }
    }
}

TEST_F(DeviceFixture, ComputePackUntilizeDst) {
    vector<vector<uint32_t> > num_tiles = {{1, 1}, {1, 2}, {2, 1}, {1, 4}, {2, 2}, {4, 1}};
    for(auto num_tile : num_tiles) {
        for (bool dst_full_sync_en : {true, false}) {
            unit_tests::compute::tilize::TestConfig test_config = {
                .dst_full_sync_en = dst_full_sync_en,
                .input_single_tile_size = 2 * 1024,
                .output_single_tile_size = 2 * 1024,
                .num_tiles_r = num_tile[0],
                .num_tiles_c = num_tile[1],
                .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                .golden_function = unit_tests::compute::gold_standard_untilize
            };
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}

//Tests pack_untilize with tiny tile dims.
//Row dim 1x32, which is faces = 2, rows = 1
//Row dim 1x16, which is faces = 1, rows = 1
TEST_F(DeviceFixture, ComputePackUntilizeDstTinyTile) {
    vector<vector<uint32_t> > test_config_values = {{1, 1, 1, 1}, {1, 1, 2, 1}, {1, 2, 2, 1}};
    uint32_t face_c_dim = 16;
    for(auto test_config_value : test_config_values) {
        for (bool dst_full_sync_en : {true, false}) {
            uint32_t num_faces_per_tile = test_config_value[2];
            uint32_t face_r_dim = test_config_value[3];
            unit_tests::compute::tilize::TestConfig test_config = {
                .short_init = true,
                .dst_full_sync_en = dst_full_sync_en,
                .input_single_tile_size = 2 * 1024,
                .output_single_tile_size = 2 * num_faces_per_tile * face_r_dim * face_c_dim,
                .num_tiles_r = test_config_value[0],
                .num_tiles_c = test_config_value[1],
                .num_faces_per_tile = num_faces_per_tile,
                .face_r_dim = face_r_dim,
                .untilize_type = unit_tests::compute::tilize::UntilizeType::DST,
                .golden_function = unit_tests::compute::gold_standard_untilize
            };
            unit_tests::compute::tilize::run_single_core_tilize_program(this->devices_.at(0), test_config);
        }
    }
}
