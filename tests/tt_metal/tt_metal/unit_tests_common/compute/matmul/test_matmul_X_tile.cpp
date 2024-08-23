// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tests/tt_metal/tt_metal/unit_tests_common/common/common_fixture.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include "test_tiles.hpp"
#include "tests/tt_metal/test_utils/tilization.hpp"
#include "tests/tt_metal/test_utils/print_helpers.hpp"
#include "tests/tt_metal/tt_metal/unit_tests_common/compute/matmul/matmul_utils.hpp"

using namespace tt;

namespace unit_tests_common::matmul::test_matmul_X_tile{

struct MatmulTileConfig {
    uint32_t M, K, N;
    bool with_bias = false;
    bool test_init_short = false;
    bool with_dt = true;
    string reader_kernel;
    string compute_kernel;
    vector<uint32_t> compute_kernel_args;
};

bool matmul_tile(CommonFixture *fixture, tt_metal::Device *device, const MatmulTileConfig &cfg, vector<uint32_t> activations, vector<std::seed_seq::result_type> weights, deprecated::Tensor<bfloat16> tensor){
    bool pass = true;

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    // num_tile == M == N == K in the case of multi_tile, conveniently they were all the same!!
    // for single_tile case, num_tile = 1
    uint32_t M = cfg.M;
    uint32_t K = cfg.K;
    uint32_t N = cfg.N;
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = M * K;             // only if M = K = N
    uint32_t dram_buffer_size = single_tile_size * num_tiles;
    // for multi_tile case buffer size will vary depending on M, N, K
    // uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    // uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    // uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels


    tt_metal::InterleavedBufferConfig dram_config{
                .device=device,
                .size = dram_buffer_size,
                .page_size = dram_buffer_size,
                .buffer_type = tt_metal::BufferType::DRAM
    };

    auto src0_dram_buffer = CreateBuffer(dram_config);
    auto src1_dram_buffer = CreateBuffer(dram_config);
    auto dst_dram_buffer = CreateBuffer(dram_config);

    uint32_t num_input_tiles = 2 * M;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    std::shared_ptr<tt_metal::Buffer> src2_dram_buffer;
    std::shared_ptr<tt_metal::Buffer> dst1_dram_buffer;
    if (cfg.with_bias) { // with_bias only when M, N, or K > 1
        tt_metal::InterleavedBufferConfig bias_config{
                    .device=device,
                    .size = single_tile_size * N,
                    .page_size = single_tile_size * N,
                    .buffer_type = tt_metal::BufferType::DRAM
        };
        src2_dram_buffer = CreateBuffer(bias_config);

        uint32_t src2_cb_index = 2;
        tt_metal::CircularBufferConfig cb_src2_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src2_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src2_cb_index, single_tile_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, core, cb_src2_config);
    } else if (cfg.test_init_short) {// This will be dummy input in uint16_t
        uint32_t in2_id = 2;
        uint32_t out1_id = 17;

        tt_metal::InterleavedBufferConfig dummy_config{
                    .device=device,
                    .size = single_tile_size * N,
                    .page_size = single_tile_size * N,
                    .buffer_type = tt_metal::BufferType::DRAM
        };

        // This will be srcB in uint16_t
        src2_dram_buffer = CreateBuffer(dummy_config);

        // This will be dummy output in uint16_t
        dst1_dram_buffer = CreateBuffer(dummy_config);

        tt_metal::CircularBufferConfig cb_src2_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{in2_id, tt::DataFormat::UInt16}})
            .set_page_size(in2_id, single_tile_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

        tt_metal::CircularBufferConfig cb_dst1_config =
        tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{out1_id, tt::DataFormat::UInt16}})
            .set_page_size(out1_id, single_tile_size);
        auto cb_dst1 = tt_metal::CreateCircularBuffer(program, core, cb_dst1_config);
    }

    uint32_t ouput_cb_index = 16;
    vector<uint32_t> reader_l1_args;
    if (cfg.M > 1 || cfg.N > 1 || cfg.K > 1){
        uint32_t intermediate_cb_index = 24;
        std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
            {ouput_cb_index, tt::DataFormat::Float16_b},
            {intermediate_cb_index, tt::DataFormat::Float16_b}
        };

        CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_tiles * single_tile_size, partials_and_out_data_format_spec)
            .set_page_size(ouput_cb_index, single_tile_size)
            .set_page_size(intermediate_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        reader_l1_args = {
            src0_dram_buffer->address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            src1_dram_buffer->address(),
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            (std::uint32_t)K,
            (std::uint32_t)M,
            (std::uint32_t)N,
            (std::uint32_t)(M * single_tile_size),
            (std::uint32_t)(N * single_tile_size),
            cfg.with_bias
        };
    } else {
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        reader_l1_args = {
            src0_dram_buffer->address(),
            (std::uint32_t)dram_src0_noc_xy.x,
            (std::uint32_t)dram_src0_noc_xy.y,
            src1_dram_buffer->address(),
            (std::uint32_t)dram_src1_noc_xy.x,
            (std::uint32_t)dram_src1_noc_xy.y,
            1,
            1,
            1,
            1 * single_tile_size,
            1 * single_tile_size
        };
    }

    std::map<string, string> compute_defines;

    if (cfg.with_dt) {
        compute_defines["WITH_DT"] = "1";
    } else {
        compute_defines["WITH_DT"] = "0";
    }
    if (cfg.test_init_short) {
        compute_defines["TEST_INIT_SHORT"] = "1";
    } else {
        compute_defines["TEST_INIT_SHORT"] = "0";
    }

    auto mm_reader_kernel = tt_metal::CreateKernel(
        program,
        cfg.reader_kernel,
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        cfg.compute_kernel,
        core,
        tt_metal::ComputeConfig{.compile_args = cfg.compute_kernel_args, .defines = compute_defines}
    );

    fixture->WriteBuffer(device, src0_dram_buffer, activations);
    fixture->WriteBuffer(device, src1_dram_buffer, weights);

    if (cfg.with_bias || cfg.test_init_short){
        vector<uint32_t> bias(N * 512, 0);
        fixture->WriteBuffer(device, src2_dram_buffer, bias);

        auto dram_src2_noc_xy = src2_dram_buffer->noc_coordinates();
        vector<uint32_t> bias_args = {
            src2_dram_buffer->address(),
            (std::uint32_t)dram_src2_noc_xy.x,
            (std::uint32_t)dram_src2_noc_xy.y,
            (std::uint32_t)N,
            (std::uint32_t)(N * single_tile_size)
        };

        for (uint32_t arg: bias_args) {
            reader_l1_args.push_back(arg);
        }
    }

    tt_metal::SetRuntimeArgs(
        program,
        mm_reader_kernel,
        core,
        reader_l1_args);

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (std::uint32_t)dram_dst_noc_xy.x,
        (std::uint32_t)dram_dst_noc_xy.y,
        num_tiles}); // this is M * N in the multi_tile case !!

    fixture->RunProgram(device, program);

    vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, dst_dram_buffer, result_vec);

    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    auto result_flat_layout = convert_to_flat_layout(result_bfp16);

    if (cfg.M > 1 || cfg.N > 1 || cfg.K > 1){
        auto result_untilized = test_utils::untilize(result_flat_layout, M*32, N*32);
        pass &= (tensor.get_values() == result_untilized);
    }else {
        pass &= (tensor.get_values() == result_flat_layout); // src1 is all 0's
    }
    DeallocateBuffer(*src0_dram_buffer);
    DeallocateBuffer(*src1_dram_buffer);
    if (cfg.with_bias || cfg.test_init_short) {
        if (cfg.test_init_short) {
            DeallocateBuffer(*dst1_dram_buffer);
        }
        DeallocateBuffer(*src2_dram_buffer);
    }
    DeallocateBuffer(*dst_dram_buffer);
    return pass;
}
} // namespace unit_tests_common::matmul::test_matmul_X_tile

TEST_F(CommonFixture, MatmulSingleTile){
    unit_tests_common::matmul::test_matmul_X_tile::MatmulTileConfig matmul_config = {
        .M = 1, .K = 1, .N = 1,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
        .compute_kernel_args = {
            1, // block_tile_dim
            1, // dst_tile_rows
            1, // dst_tile_cols
            1, // block_cnt
            1, // in0_block_tile_cnt
            1, // in1_block_tile_cnt
            1 // out_block_tile_cnt
        }
    };
    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tile_layout = convert_to_tile_layout(tensor.get_values());
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);

    auto identity = create_identity_matrix(32, 32, 32); //bfloat16 32x32 identity
    auto weights_tile_layout = convert_to_tile_layout(identity);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);

    for(unsigned int id = 0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_X_tile::matmul_tile(this, devices_.at(id), matmul_config, activations, weights, tensor));
    }
}

TEST_F(CommonFixture, MatmulMultiTile){
    uint32_t M = 4;
    uint32_t N = 4;
    uint32_t K = 4;
    unit_tests_common::matmul::test_matmul_X_tile::MatmulTileConfig matmul_config = {
        .M = M, .K = K, .N = N,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_with_bias.cpp",
        .compute_kernel_args = {
            1, // block_tile_dim, within block, how many tiles are on the K dim
            M, // dst_tile_rows
            N, // dst_tile_cols
            K, // block_cnt, across blocks, how many tiles are on the K dim
            M, // in0_block_tile_cnt, M * block_tile_dim
            N, // in1_block_tile_cnt,  N * block_tile_dim
            (M * N), // out_block_tile_cnt
            matmul_config.with_bias // whether or not to use bias
        }
    };

    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    auto activations_tile_transposed = transpose_tiles(activations, M, K, 1);

    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bfloat16 32x32 identity
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);

    for(unsigned int id = 0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_X_tile::matmul_tile(this, devices_.at(id), matmul_config, activations_tile_transposed, weights, tensor));
        log_info(LogTest, "Multi tile with no bias passed");
        matmul_config.with_bias = true;
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_X_tile::matmul_tile(this, devices_.at(id), matmul_config, activations_tile_transposed, weights, tensor));
        log_info(LogTest, "Multi tile with bias passed");
    }
}

TEST_F(CommonFixture, MatmulBlock){
    uint32_t M = 4;
    uint32_t N = 4;
    uint32_t K = 4;
    unit_tests_common::matmul::test_matmul_X_tile::MatmulTileConfig matmul_config = {
        .M = M, .K = K, .N = N,
        .test_init_short = false,
        .with_dt = false,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp",
        .compute_kernel_args = {
            1, // block_tile_dim, within block, how many tiles are on the K dim
            M, // dst_tile_rows
            N, // dst_tile_cols
            K, // block_cnt, across blocks, how many tiles are on the K dim
            M, // in0_block_tile_cnt, M * block_tile_dim
            N, // in1_block_tile_cnt,  N * block_tile_dim
            (M * N), // out_block_tile_cnt
        }
    };

    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    auto activations_tile_transposed = transpose_tiles(activations, M, K, 1);

    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bfloat16 32x32 identity
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);

    for(unsigned int id = 0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_X_tile::matmul_tile(this, devices_.at(id), matmul_config, activations_tile_transposed, weights, tensor));
    }
}

TEST_F(CommonFixture, MatmulBlockInitShort){
    uint32_t M = 4;
    uint32_t N = 4;
    uint32_t K = 4;
    unit_tests_common::matmul::test_matmul_X_tile::MatmulTileConfig matmul_config = {
        .M = M, .K = K, .N = N,
        .test_init_short = true,
        .with_dt = false,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp",
        .compute_kernel_args = {
            1, // block_tile_dim, within block, how many tiles are on the K dim
            M, // dst_tile_rows
            N, // dst_tile_cols
            K, // block_cnt, across blocks, how many tiles are on the K dim
            M, // in0_block_tile_cnt, M * block_tile_dim
            N, // in1_block_tile_cnt,  N * block_tile_dim
            (M * N), // out_block_tile_cnt
        }
    };

    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    auto activations_tile_transposed = transpose_tiles(activations, M, K, 1);

    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bfloat16 32x32 identity
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);

    for(unsigned int id = 0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_X_tile::matmul_tile(this, devices_.at(id), matmul_config, activations_tile_transposed, weights, tensor));
    }
}

TEST_F(CommonFixture, MatmulBlockInitShortWithDt){
    uint32_t M = 4;
    uint32_t N = 4;
    uint32_t K = 4;
    unit_tests_common::matmul::test_matmul_X_tile::MatmulTileConfig matmul_config = {
        .M = M, .K = K, .N = N,
        .test_init_short = true,
        .with_dt = true,
        .reader_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        .compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block.cpp",
        .compute_kernel_args = {
            1, // block_tile_dim, within block, how many tiles are on the K dim
            M, // dst_tile_rows
            N, // dst_tile_cols
            K, // block_cnt, across blocks, how many tiles are on the K dim
            M, // in0_block_tile_cnt, M * block_tile_dim
            N, // in1_block_tile_cnt,  N * block_tile_dim
            (M * N), // out_block_tile_cnt
        }
    };

    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
    auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
    auto activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    auto activations_tile_transposed = transpose_tiles(activations, M, K, 1);

    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bfloat16 32x32 identity
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);

    for(unsigned int id = 0; id < devices_.size(); id++){
        ASSERT_TRUE(unit_tests_common::matmul::test_matmul_X_tile::matmul_tile(this, devices_.at(id), matmul_config, activations_tile_transposed, weights, tensor));
    }
}
