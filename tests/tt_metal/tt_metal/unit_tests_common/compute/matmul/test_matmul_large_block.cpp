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
using namespace tt::test_utils;

namespace unit_tests_common::matmul::test_matmul_large_block {

void set_math_fid_masks(uint16_t &math_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    auto arch = get_arch_from_string(get_env_arch_name());
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: { break; }
        case MathFidelity::HiFi2:
        case MathFidelity::LoFi: { math_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE; break; }
        default: { TT_THROW("Unsupported MathFidelity={}", math_fidelity); break; }
    }
}

void create_CBs_for_fused_matmul(tt_metal::Program &program, tt_metal::Device* device, CoreCoord core, bool activations_rm, bool output_rm, uint32_t M, uint32_t N, uint32_t in0_block_w, uint32_t out_subblock_h) {

    uint32_t num_bytes_for_df = 2;

    uint32_t in0_cb                                   = 0;
    uint32_t in1_cb                                   = 1;
    uint32_t tilize_mode_tilized_in0_cb               = 24;
    uint32_t matmul_partials_cb                       = 25;
    uint32_t untilize_mode_final_matmul_partials_cb   = 26;
    uint32_t untilize_mode_reblock_cb                 = 27;
    uint32_t out0_cb                                  = 16;

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;
    CoreRangeSet cores(std::set<CoreRange>{CoreRange(core, core)});

    // Invariants
    uint32_t cb0_tiles = M * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_in0_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{in0_cb, tt::DataFormat::Float16_b}})
        .set_page_size(in0_cb, single_tile_size);
    auto cb_in0 = tt_metal::CreateCircularBuffer(program, core, cb_in0_config);

    uint32_t cb1_tiles = N * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_in1_config = tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{in1_cb, tt::DataFormat::Float16_b}})
        .set_page_size(in1_cb, single_tile_size);
    auto cb_in1 = tt_metal::CreateCircularBuffer(program, core, cb_in1_config);

    std::map<uint8_t, tt::DataFormat> partials_and_out_data_format_spec = {
        {matmul_partials_cb, tt::DataFormat::Float16_b},
        {out0_cb, tt::DataFormat::Float16_b}
    };

    if (not activations_rm and not output_rm) { // no tilize, no untilize
        // Partials share same L1 address space as output
        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_data_format_spec)
            .set_page_size(matmul_partials_cb, single_tile_size)
            .set_page_size(out0_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, cores, cb_matmul_partials_config);

    } else if (not activations_rm and output_rm) { // no tilize, just untilize

        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N; // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config = tt_metal::CircularBufferConfig(reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        auto cb_reblock = tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(out0_cb, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (activations_rm and not output_rm) { // just tilize, no untilize

        tt_metal::CircularBufferConfig cb_src0_tilized_config = tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_data_format_spec)
            .set_page_size(matmul_partials_cb, single_tile_size)
            .set_page_size(out0_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

    } else { // tilize activations and untilize output

        // Used for placing tilized activations
        tt_metal::CircularBufferConfig cb_src0_tilized_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        auto cb_src0_tilized = tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(matmul_partials_cb, single_tile_size);
        auto cb_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Shares same address space as matmul partials
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        auto cb_final_matmul_partials = tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N; // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config = tt_metal::CircularBufferConfig(reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
            .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        auto cb_reblock = tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(out0_cb, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    }
}

bool matmul_large_block(CommonFixture *fixture, tt_metal::Device *device, bool activations_rm, bool output_rm, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    bool pass = true;

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};
    uint32_t M = 4;
    uint32_t K = 2;
    uint32_t N = K;
    int out_subblock_h = 2;
    int out_subblock_w = 1;
    int in0_block_w = K;

    uint32_t single_tile_size = 2 * 1024;
    TT_FATAL(M * in0_block_w * single_tile_size * 2 <= 150*1024);
    TT_FATAL(N * in0_block_w * single_tile_size * 2 <= 100*1024);
    TT_FATAL(M * N * single_tile_size <= 600*1024);
    uint32_t dram_buffer_size_act = single_tile_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_weights = single_tile_size * K * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_size_out = single_tile_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    tt_metal::InterleavedBufferConfig act_config{
                .device=device,
                .size = dram_buffer_size_act,
                .page_size = dram_buffer_size_act,
                .buffer_type = tt_metal::BufferType::DRAM
    };
    tt_metal::InterleavedBufferConfig weights_config{
                .device=device,
                .size = dram_buffer_size_weights,
                .page_size = dram_buffer_size_weights,
                .buffer_type = tt_metal::BufferType::DRAM
    };
    tt_metal::InterleavedBufferConfig dst_config{
                .device=device,
                .size = dram_buffer_size_out,
                .page_size = dram_buffer_size_out,
                .buffer_type = tt_metal::BufferType::DRAM
    };


    auto src0_dram_buffer = CreateBuffer(act_config);
    auto src1_dram_buffer = CreateBuffer(weights_config);
    auto dst_dram_buffer = CreateBuffer(dst_config);

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();
    auto dram_src1_noc_xy = src1_dram_buffer->noc_coordinates();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    std::vector<uint32_t> mm_reader_rt_args{
        src0_dram_buffer->address(),
        (std::uint32_t)dram_src0_noc_xy.x,
        (std::uint32_t)dram_src0_noc_xy.y,
        src1_dram_buffer->address(),
        (std::uint32_t)dram_src1_noc_xy.x,
        (std::uint32_t)dram_src1_noc_xy.y,
        (std::uint32_t)(K/in0_block_w), // num_blocks
        M * in0_block_w, // input 0 block num tiles
        N * in0_block_w, // input 1 block num tiles
        M * in0_block_w * single_tile_size, // input 0 block bytes
        N * in0_block_w * single_tile_size}; // input 1 block bytes


    std::vector<uint32_t> writer_rt_args;
    string writer_kernel;
    if (output_rm) {
        writer_kernel = "tt_metal/kernels/dataflow/writer_unary.cpp";
        writer_rt_args = {
            dst_dram_buffer->address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            uint(M * N)
        };
    } else {
        writer_kernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp";
        writer_rt_args = {
            dst_dram_buffer->address(),
            (std::uint32_t)dram_dst_noc_xy.x,
            (std::uint32_t)dram_dst_noc_xy.y,
            (std::uint32_t)out_subblock_h, // num tiles per sub block m
            (std::uint32_t)out_subblock_w, // num tiles per sub block n
            (std::uint32_t)M/out_subblock_h, // num sub blocks m
            (std::uint32_t)N/out_subblock_w, // num sub blocks n
            (std::uint32_t)out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row within sub-block
            (std::uint32_t)out_subblock_h * out_subblock_w * single_tile_size * (N/out_subblock_w), // bytes offset to next row of sub-blocks
            (std::uint32_t)out_subblock_w*single_tile_size
        }; // bytes offset to next sub-block
    }

    auto mm_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        writer_kernel,
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    int num_blocks = (K/in0_block_w);

    int in0_num_subblocks = (M/out_subblock_h);
    int in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    int in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    int in1_num_subblocks = (N/out_subblock_w);
    int in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    int in1_per_core_w = out_subblock_w * in1_num_subblocks;

    int out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    int in0_subblock_h = (in0_block_num_tiles / in0_num_subblocks) / in0_block_w;

    create_CBs_for_fused_matmul(program, device, core, activations_rm, output_rm, M, N, in0_block_w, out_subblock_h);

    TT_FATAL(in0_subblock_h * in0_block_w * in0_num_subblocks == in0_block_num_tiles);
    TT_FATAL(in0_block_w == K);

    vector<uint32_t> compute_kernel_args = {
        uint(in0_block_w),
        uint(in0_num_subblocks),
        uint(in0_block_num_tiles),
        uint(in0_subblock_num_tiles),
        uint(in0_subblock_h),

        uint(in1_num_subblocks),
        uint(in1_block_num_tiles),
        uint(in1_per_core_w),

        uint(num_blocks),

        uint(out_subblock_h),
        uint(out_subblock_w),
        uint(out_subblock_num_tiles),

        uint(activations_rm),
        uint(output_rm)
    };

    string compute_kernel = "tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block.cpp";

    auto mm_kernel = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .compile_args = compute_kernel_args
        }
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    SHAPE shape = {1, 1, M * 32, K * 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> activations;
    if (activations_rm) {
        activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
    } else {
        auto activations_tilized = test_utils::tilize(tensor.get_values(), M * 32, K * 32);
        auto activations_tile_layout = convert_to_tile_layout(activations_tilized);
        activations = pack_bfloat16_vec_into_uint32_vec(activations_tile_layout);
    }
    fixture->WriteBuffer(device, src0_dram_buffer, activations);

    auto identity = create_identity_matrix(K * 32, N * 32, std::min(K, N) * 32); //bflaot16 32x32 identity
    auto identity_tilized = test_utils::tilize(identity, K * 32, N * 32);
    auto weights_tile_layout = convert_to_tile_layout(identity_tilized);
    auto weights = pack_bfloat16_vec_into_uint32_vec(weights_tile_layout);
    fixture->WriteBuffer(device, src1_dram_buffer, weights);

    tt_metal::SetRuntimeArgs(
        program,
        mm_reader_kernel,
        core,
        mm_reader_rt_args);

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel,
        core,
        writer_rt_args);

    CoreCoord debug_core = {1, 1};

    fixture->RunProgram(device, program);
    std::vector<uint32_t> result_vec;
    fixture->ReadBuffer(device, dst_dram_buffer, result_vec);

    ////////////////////////////////////////////////////////////////////////////
    //                      Validation & Teardown
    ////////////////////////////////////////////////////////////////////////////
    auto result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    auto golden = tensor.get_values();
    uint16_t math_fid_mask = 0xFFFF;
    set_math_fid_masks(math_fid_mask, math_fidelity);
    // If we're testing LoFi/HiFi2 we generate matching golden (trunc LSB).
    // Note that this will work only for multiplying with identity matrix
    for (auto i = 0; i < golden.size(); i++) {
        golden[i] = bfloat16(golden[i].to_uint16() & math_fid_mask);
    }


    if (output_rm) {
        pass &= (golden == result_bfp16);
        if (not pass) {
            print_faces(result_bfp16, "Result");
        }
    } else {
        auto result_flat_layout = convert_to_flat_layout(result_bfp16);
        auto result_untilized = test_utils::untilize(result_flat_layout, M*32, N*32);
        pass &= (golden == result_untilized);
        if (not pass) {
            print_faces(result_untilized, "Result");
        }
    }

    if (not pass) {
        print_faces(tensor.get_values(), "Golden");
    }
    return pass;
}

}

TEST_F(CommonFixture, MatmulLargeBlock) {
    for (uint8_t i = uint8_t(MathFidelity::LoFi); i <= uint8_t(MathFidelity::HiFi4); i++) {
        if (i == 1) continue;;
        tt::log_info(tt::LogTest, "Math Fidelity = {}", i);
        for (unsigned int id=0; id < devices_.size(); id++){
            ASSERT_TRUE(unit_tests_common::matmul::test_matmul_large_block::matmul_large_block(this, devices_.at(id), false, false, MathFidelity(i)));
            log_info (LogTest, "Tilized input, Tilized output Passed");
            ASSERT_TRUE(unit_tests_common::matmul::test_matmul_large_block::matmul_large_block(this, devices_.at(id), true, false, MathFidelity(i)));
            log_info (LogTest, "Row major input, Tilized output Passed");
            ASSERT_TRUE(unit_tests_common::matmul::test_matmul_large_block::matmul_large_block(this, devices_.at(id), false, true, MathFidelity(i)));
            log_info (LogTest, "Tilized input, Row major output Passed");
            ASSERT_TRUE(unit_tests_common::matmul::test_matmul_large_block::matmul_large_block(this, devices_.at(id), true, true, MathFidelity(i)));
            log_info (LogTest, "Row major input, Row major output Passed");
        }
    }
}
